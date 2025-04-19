wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2-small-FPN-augment" \
    --model "dinov2" \
    --weight-decay 0.0001 \
    --onecycle \
    # --multistep \
    # --scheduler-epochs 40 60 80 \
    # --fine-tune \
    # --weighted \
    