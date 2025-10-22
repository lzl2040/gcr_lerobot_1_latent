python lerobot/scripts/merge_lora.py \
    --policy.type="pi0" \
    --policy.use_lora=true \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix="pizza_v9_task_20_v0" \
    --dataset.image_transforms.enable=true \
    --wandb.enable=false \
    --wandb.project="pi0-ft-simulated" \
    --job_name="pi0-04-21-ft-vlabench-local-bs-128-cos-sche" \
    --log_dir="logs" \
    --output_dir="0421-ft-vlabench-bs-128-1st-cos-sche" \
    --steps=30_000 \
    # --dataset.image_transforms.enable=true
    # --deepspeed="./ds_zero2.json" \