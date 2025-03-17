wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2-training" \
    --model "dinov2" \
    --scheduler \
    --scheduler-epochs 30 60 90 \