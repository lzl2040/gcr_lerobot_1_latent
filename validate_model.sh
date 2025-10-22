CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1 lerobot/scripts/validate_model.py \
    --deepspeed="./ds_zero2_40G.json" \
    --policy.type="pi0" \
    --policy.use_lora=false \
    --dataset.root="/Data/lerobot_data/real_world" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix="american_data" \
    --dataset.use_state=true \
    --dataset.image_transforms.enable=false \
    --wandb.enable=false \
    --wandb.project="pi0-ft-simulated" \
    --job_name="08-19-pt-pi0-ft-pizza-v9-task-16-sep1-chunk-12" \
    --log_dir="logs" \
    --output_dir="/Data/lzl/pi0-ft-simulated/08-19-pt-pi0-ft-pizza-v9-task-16-sep1-chunk-12" \
    --steps=30_000 \
    # --policy.pretrained_path="/Data/lzl/pi0-ft-real/1015-american-data-w-state/mp_rank_00_model_states.pt" \
    # --dataset.image_transforms.enable=true