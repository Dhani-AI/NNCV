wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 20 \
    --epochs 65 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2-training" \
    --model "dinov2" \
    --scheduler \
    --scheduler-epochs 50 \