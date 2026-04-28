import torch

from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from typing import Optional
from torch import tanh
from .grit_layer import GritTransformerLayer

from yacs.config import CfgNode as CN # only for grit transformer config.

from models.flat_nsd import FlatBundleConv

class GNN(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        node_level_task: bool = False,
        conv_layer: str = "GCNConv",
        alpha: Optional[float] = None,
        activ_fun: str = "tanh",
        dropout_prob: float = 0.0,
        edge_dim: Optional[int] = None,
        stalk_dimension: Optional[int] = None,
        epsilon: Optional[float] = 0.1,
        gamma: Optional[float] = 0.1,
        backbone_hidden: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        **kwargs
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.edge_dim = edge_dim

        if backbone_hidden is None:
            self.backbone_hidden = hidden_dim
        else:
            self.backbone_hidden = backbone_hidden
        
        if backbone_layers is None:
            self.backbone_layers = 1
        else:
            self.backbone_layers = backbone_layers

        self.d = stalk_dimension if stalk_dimension is not None else 1

        try:
            self.activation = getattr(torch, activ_fun)
        except:  
            self.activation = getattr(torch, "relu")
            print(f"Activation function {activ_fun} not found. Using default")
        
        self.emb = Linear(self.input_dim, self.hidden_dim if conv_layer != "FlatNSD" else self.hidden_dim * self.d)
        
        self.edge_emb = None
        if conv_layer == 'GRIT' and self.edge_dim is not None:
            self.edge_emb = Linear(self.edge_dim, self.hidden_dim)

        self.conv_layer = getattr(pyg_nn, conv_layer) if conv_layer not in ['GRIT', 'FlatNSD'] else GritTransformerLayer
        self.conv_name = conv_layer
        self.conv = ModuleList()

        for _ in range(num_layers):
            if conv_layer == "GINConv":
                mlp = Linear(self.hidden_dim, self.hidden_dim)
                self.conv.append(self.conv_layer(nn=mlp, train_eps=True)) #type: ignore
            elif conv_layer == "GCN2Conv":
                self.conv.append(self.conv_layer(channels=self.hidden_dim, 
                                                 alpha=self.alpha)) #type: ignore
            elif conv_layer == "GPSConv":
                attn_kwargs = {"dropout": 0.0}
                nn = pyg_nn.GCNConv(self.hidden_dim, self.hidden_dim) # todo, test gps with gineconv
                self.conv.append(
                    pyg_nn.GPSConv(
                        self.hidden_dim,
                        nn,
                        heads=2,
                        # attn_type='multihead', attn_kwargs=attn_kwargs, # pyg >= 2.4
                        # attn_dropout=0.0, # pyg < 2.4.0
                        norm="layer",
                    )
                )
            elif conv_layer == "Baseline":
                self.conv.append(BaselineMLP(self.hidden_dim, self.hidden_dim))

            elif conv_layer == "GINEConv":
                self.conv.append(self.conv_layer(
                    nn=Linear(self.hidden_dim, self.hidden_dim),
                    train_eps=True,
                    edge_dim=self.edge_dim if self.edge_dim is not None else 2, 
                )) #type: ignore

            elif conv_layer == "GRIT":
                grit_num_heads = kwargs.get("grit_num_heads", 4)
                grit_attn_dropout = kwargs.get("grit_attn_dropout", 0.0)
                grit_act = kwargs.get("grit_act", "relu")

                cfg = CN()
                # add any specific GRIT configurations here if needed
                cfg.attn = CN()
                cfg.attn.clamp = 5.
                cfg.attn.act = grit_act
                cfg.attn.full_attn = True
                cfg.attn.edge_enhance = True
                cfg.attn.O_e = True
                cfg.attn.norm_e = True
                cfg.attn.fwl = False
                cfg.bn_momentum = 0.1
                cfg.bn_no_runner = False
                cfg.dropout = dropout_prob
                cfg.num_heads = grit_num_heads

        
                self.conv.append(self.conv_layer(
                    in_dim=self.hidden_dim, 
                    out_dim=self.hidden_dim, 
                    num_heads=grit_num_heads,
                    dropout=dropout_prob,
                    attn_dropout=grit_attn_dropout,
                    act=grit_act,
                    cfg=cfg
                ))
            elif conv_layer == "FlatNSD":
                self.conv.append(FlatBundleConv(in_channels=self.hidden_dim,
                                  out_channels=self.hidden_dim,
                                  stalk_dimension=self.d,
                                  dropout=dropout_prob,
                                  linear_emb=True,
                                  gnn_type='SAGE',
                                  gnn_layers=self.backbone_layers,
                                  gnn_hidden=self.backbone_hidden,
                                  epsilon=epsilon,
                                  gamma=gamma))
            else:
                self.conv.append(self.conv_layer(in_channels=self.hidden_dim, 
                                                 out_channels=self.hidden_dim)) #type: ignore


        self.node_level_task = node_level_task
        if self.node_level_task:
            self.readout = Sequential(
                Linear(self.hidden_dim if conv_layer != "FlatNSD" else self.hidden_dim * self.d, self.hidden_dim // 2),
                LeakyReLU(),
                Linear(self.hidden_dim // 2, self.output_dim)
            )
        else:
            self.readout = Sequential(
                Linear(self.hidden_dim * 3 if conv_layer != "FlatNSD" else self.hidden_dim * 3 * self.d, (self.hidden_dim * 3) // 2),
                LeakyReLU(),
                Linear((self.hidden_dim * 3) // 2, self.output_dim)
            )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.emb(x) if self.emb else x

        if self.conv_name == "GRIT":
            data_cloned = data.clone()
            data_cloned.x = x
            # ad an attribute "E" to data_cloned for edge features
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                if self.edge_emb is not None:
                    data_cloned.edge_attr = self.edge_emb(data.edge_attr)
                else:
                    data_cloned.edge_attr = data.edge_attr
            else:
                data_cloned.edge_attr = None

        
        x_0 = None
        if self.conv_name == "GCN2Conv":
            x_0 = x

        for i, conv in enumerate(self.conv):
            x_prev = x
            if self.conv_name == "GCN2Conv":
                x = conv(x, x_0, edge_index)
            elif self.conv_name == "GINEConv":
                x = conv(x, edge_index, data.edge_attr)
            elif self.conv_name == "GRIT":
                data_cloned = conv(data_cloned) #type: ignore
            else: 
                x = conv(x, edge_index)
            
            x = self.activation(x)
            x = self.dropout(x)
            
        if self.conv_name == "GRIT":
            x = data_cloned.x #type: ignore
        if not self.node_level_task:
            x = torch.cat(
                [
                    global_add_pool(x, batch), #type: ignore
                    global_max_pool(x, batch), #type: ignore
                    global_mean_pool(x, batch), #type: ignore
                ],
                dim=1,
            )

    
        x = self.readout(x)
        return x
