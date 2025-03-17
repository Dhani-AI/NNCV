wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 10 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training" \
    --model "dino" \