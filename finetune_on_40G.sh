USE_STATE=true
JOB_NAME="1009-american-data-w-state"
DATA_MIX="simpler"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_state)
            USE_STATE="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --data_mix)
            DATA_MIX="$2"
            shift 2
            ;;
    esac
done
OUTPUT_DIR="/mnt/wangxiaofa/latent-ft-simulated/${JOB_NAME}"

python lerobot/scripts/dps_train.py \
    --deepspeed="./ds_zero2_40G.json" \
    --policy.type="pi0" \
    --policy.use_lora=false \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix=$DATA_MIX \
    --dataset.use_state=$USE_STATE \
    --dataset.image_transforms.enable=false \
    --wandb.enable=false \
    --resume=true \
    --wandb.project="latent-ft-simulated" \
    --job_name=$JOB_NAME \
    --log_dir="/mnt/wangxiaofa/logs" \
    --output_dir=$OUTPUT_DIR \
    --steps=300_000 \
    --save_freq=5000 \
    --policy.chunk_size=15 \
    --policy.n_action_steps=15 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=true \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=2000 \
    --policy.scheduler_decay_steps=40000 \
    --policy.pt_weight_path="/mnt/wangxiaofa/latent_action_exp/1019_latent_action_distill_mse_loss/step40000.pt"