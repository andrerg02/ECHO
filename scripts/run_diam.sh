# python scripts/train.py \
#     --task diam \
#     --gnn_type GNN \
#     --conv_layer FlatNSD  \
#     --hidden_dim 137 \
#     --num_layers 25 \
#     --lr 0.0009331582891477268 \
#     --batch_size 128 \
#     --weight_decay 0.0004385659794959314 \
#     --stalk_dimension 5 \
#     --epsilon 0.4937340024806669 \
#     --gamma 0.28144239564760987

python scripts/train.py \
    --task diam \
    --gnn_type GNN \
    --conv_layer FlatNSD  \
    --hidden_dim 152 \
    --num_layers 27 \
    --lr 0.0005441816486669067 \
    --batch_size 128 \
    --weight_decay 0.00045298390534858355 \
    --stalk_dimension 5 \
    --backbone_hidden 32 \
    --backbone_layers 5