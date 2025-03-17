wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 20 \
    --epochs 50 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2-training" \
    --model "dinov2" \