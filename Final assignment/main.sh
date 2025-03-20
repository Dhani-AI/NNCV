wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2-FineTuning" \
    --model "dinov2" \
    --scheduler \
    --scheduler-epochs 80 \