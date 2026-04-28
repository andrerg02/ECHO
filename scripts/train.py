## add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Task to run: [sssp, ecc, diam, chem]",)

parser.add_argument("--device", type=str, default="gpu", help="Device to use for training")
# general gnn parameters
parser.add_argument("--conv_layer", type=str)
parser.add_argument("--num_layers", type=int, help="Number of layers in the GNN")
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of the GNN")
parser.add_argument("--lr", type=float, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader")
parser.add_argument("--gnn_type", type=str)

# sheaf specific parameters
parser.add_argument("--stalk_dimension", type=int, help="Stalk dimension for the FlatNSD model")
parser.add_argument("--backbone_hidden", type=int, help="Hidden dimension for the backbone GNN in the FlatNSD model")
parser.add_argument("--backbone_layers", type=int, help="Number of layers in the backbone GNN in the FlatNSD model")

# adgn, swan specific params
parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the ADGN model")
parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for the ADGN model")
parser.add_argument("--activ_fun", type=str, default="tanh", help="Activation function for the ADGN model")
parser.add_argument("--graph_conv", type=str, default="GCNConv", help="Graph convolution layer for the ADGN model")
parser.add_argument("--bias", type=bool, help="Use bias in the ADGN model")
parser.add_argument("--train_weights", type=bool)
parser.add_argument("--weight_sharing", type=bool, help="Use weight sharing in the ADGN model")

# drew specific parameters
parser.add_argument("--khop", type=int)
parser.add_argument("--delay", type=bool)
parser.add_argument("--constant_feature", type=float, help="Constant feature")

# gcn2 params
parser.add_argument("--alpha", type=float, help="Alpha for the GCN2 model")

# phdgn specific parameters
parser.add_argument("--beta", type=float, help="Beta parameter for the PHDGN model")
parser.add_argument("--p_conv_mode", type=str, choices=["naive", "gcn"], help="P convolution mode for the PhDGN model")
parser.add_argument("--q_conv_mode", type=str, choices=["naive", "gcn"], help="Q convolution mode for the PhDGN model")
parser.add_argument("--doubled_dim", type=bool, choices=[True, False], help="Whether to double the dimension in the PhDGN model")
parser.add_argument("--final_state", type=str, choices=["p", "q", "pq"], help="Final state mode for the PhDGN model")
parser.add_argument("--dampening_mode", type=str, choices=["param", "param+", "MLP4ReLU", "DGNReLU", "none"], help="Dampening mode for the PhDGN model")
parser.add_argument("--external_mode", type=str, choices=["MLP4Sin", "DGNtanh", "none"], help="External mode for the PhDGN model")


from torch_geometric.loader import DataLoader

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import os
from utils import get_dataset, KHopTransform
from utils.litmodels import LitGraphNN
import re


torch.set_float32_matmul_precision("high")
get_epoch = lambda path: int(re.findall(r"epoch=(\d+)", path)[0])


def train(seed, config):
    """Train and validate the model"""
    config = parser.parse_args()
    task = config.task

    L.seed_everything(seed) 
    batch_size = config.batch_size

    print("Current directory: ", os.getcwd())

    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root="./data/",
        task=task,
        pre_transform=(
            KHopTransform(k=config.k_hop)
            if config.gnn_type == "DRew_GCN"
            else None
        ),
        constant_feature=config.constant_feature,
    )

    scaling_factor = data_train.scaling_factor[task]

    if scaling_factor is None and task == "chem":
        scaling_factor = 1.0


    print(f"Scaling factor for {task}: {scaling_factor}")
    print(f"Scaling factor: {scaling_factor}")

    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )


    print("Data loaded")

    hp_conf = vars(config)

    model = LitGraphNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=False if task == "diam" else True,
        scaling_factor=scaling_factor,
        **hp_conf,
    )

    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=100),
            ModelCheckpoint(monitor="val_loss", save_top_k=1),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    best_epoch = get_epoch(trainer.checkpoint_callback.best_model_path) #type: ignore
    print(f"Best epoch: {best_epoch}")
    print("Best checkpoint path: ", trainer.checkpoint_callback.best_model_path) #type: ignore

    trainer.validate(model, val_loader, ckpt_path="best")
    print("Testing model")
    print(test_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    # log metrics to a dictionary and return it.
    metrics = {
        "train_loss": trainer.callback_metrics["train_loss"].item(),
        "val_loss": trainer.callback_metrics["val_loss"].item(),
        "val_mse": trainer.callback_metrics["val_mse"].item(),
        "val_mae": trainer.callback_metrics["val_mae"].item(),
        "test_loss": trainer.callback_metrics["test_loss"].item(),
        "test_mse": trainer.callback_metrics["test_mse"].item(),
        "test_mae": trainer.callback_metrics["test_mae"].item(),
        "test_acc": trainer.callback_metrics["test_acc"].item(),
        "best_epoch": best_epoch,
        "best_checkpoint_path": trainer.checkpoint_callback.best_model_path, #type: ignore
    }

    return metrics



if __name__ == "__main__":
    args = parser.parse_args()
    metrics = train(
        seed=43,
        config=args,
    )

    print("Metrics: ", metrics)
    
