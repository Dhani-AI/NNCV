wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 150 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2-finetune-small-32" \
    --model "dinov2" \
    --weight-decay 0.0001 \
    --multistep \
    --scheduler-epochs 80 \
    --fine-tune \
    # --onecycle \
    # --weighted \
    