#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
import os
import json
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import deepspeed
from deepspeed import get_accelerator

import torch
from termcolor import colored
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining, LeRobotDataset, MultiSameDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy



def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if cfg.local_rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {cfg.local_rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # # 控制台Handler
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        
        # 文件Handler
        log_path = Path(cfg.log_dir) / f"logs_with_pretrain/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def update_policy(
    model_engine,
    batch: Any,
) -> tuple[MetricsTracker, dict]:
    
    batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    torch.cuda.empty_cache()
    loss, output_dict = model_engine(batch)

    model_engine.backward(loss)
    model_engine.step()
    return loss, output_dict


def count_parameters_mb(model, logger):
    total_params = 0
    trainable_params = 0
    lora_params = 0
    action_expert_params = 0
    vision_encoder_params = 0

    for name, param in model.named_parameters():
        param_size = param.numel()  # size in bytes
        if "gemma_expert" in name:
            action_expert_params += param_size
        
        if "vision_tower" in name:
            vision_encoder_params += param_size
        if param.requires_grad:
            trainable_params += param_size
            if "lora" in name.lower():  # LoRA 层通常包含 "lora" 关键字
                lora_params += param_size
        
        
        total_params += param_size

    def to_billion(n_params):
        return n_params / 1e9

    logger.info(f"📦 action_expert 参数数量: {to_billion(action_expert_params):.3f} B")
    logger.info(f"🎯 可训练参数数量: {to_billion(trainable_params):.3f} B")
    logger.info(f"🔧 LoRA 参数数量: {to_billion(lora_params):.3f} B")
    logger.info(f"📷 视觉编码器参数数量: {to_billion(vision_encoder_params):.3f} B")
    logger.info(f"📊 模型总参数数量: {to_billion(total_params):.3f} B")

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    # Initialize DeepSpeed
    # deepspeed.init_distributed()
    logger = init_logger(cfg)
    
    # image_transforms = (
    #     ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    # )
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    # wrist_image_transforms = (
    #     ImageTransforms(cfg.dataset.wrist_image_transforms) if cfg.dataset.image_transforms.enable else None
    # )
    print(f"Image transforms:{image_transforms}")
    # print(wrist_image_transforms)
    
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + int(os.environ.get('RANK', 0)))

    # Dataset setup
    # dataset = MultiDatasetforDistTraining(cfg=cfg, image_transforms=image_transforms, 
    #                        seed=cfg.seed, 
    #                        # data_mix="oxe_magic_soup_plus",
    #                        data_mix=cfg.dataset.data_mix,
    #                        vla2root_json="vla2root.json")
    # for finetuning on simuleted enviroments
    # data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format/libero_spatial_no_noops_lerobot"
    # dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, 
    #                          root=cfg.dataset.root,
    #                          image_transforms=image_transforms)
    # Single Dataset
    # dataset = make_dataset(cfg)
    # data_names = ["libero_spatial_no_noops_lerobot", "libero_goal_no_noops_lerobot",
    #               "libero_object_no_noops_lerobot", "libero_10_no_noops_lerobot"]
    # logger.info(f"Dataset names:{data_names}")
    dataset = MultiSameDataset(cfg=cfg, 
                               image_transforms=image_transforms,
                            #    wrist_image_transforms=wrist_image_transforms)
    )
    
    logger.info(f"Data load from:{cfg.dataset.root}")
    logger.info(f"Dataset: {dataset}")

    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setting model's tokenizer_max_length to 100")
        cfg.policy.tokenizer_max_length=100
    logger.info("Still creating policy...")
    # data_root = "/mnt/wangxiaofa/pi0-ft-simulated/0723-ft-pizza-v9-task-20-v0-chunk-12-wo-state-lora-bs-4-8gpu-gra-acc-2-with-lr-decay-warm-1k-wd-1e-2-normal-lr-aug-1st"
    # # file_list = sorted(os.listdir(data_root))
    # # print(f"File list:{file_list}")
    # for ckt_id in range(10000, 20000, 1000):
    #     ckt_dir = os.path.join(data_root, f"global_step{ckt_id}")
    #     logger.info(f"Load checkpoint from:{ckt_dir}")
    #     ckt_path = os.path.join(ckt_dir, "mp_rank_00_model_states.pt")
    #     # print(cfg.policy.pretrained_path)
    #     policy = make_policy(
    #         cfg=cfg.policy,
    #         device='cpu',
    #         ds_meta=dataset.meta,
    #         weight_pt_path=ckt_path
    #     )
        
    #     logger.info("Policy model created...")
    #     count_parameters_mb(policy, logger)
    #     print("Applying the LoRA")
    #     lora_paligemma = policy.model.paligemma_with_expert.paligemma.merge_and_unload()
    #     # print(lora_paligemma)
    #     policy.model.paligemma_with_expert.paligemma = lora_paligemma
    #     save_path = os.path.join(ckt_dir, "lora_merge.pt")
    #     torch.save(policy.state_dict(), save_path)
    #     print("Save")
    
    policy = make_policy(
        cfg=cfg.policy,
        device='cpu',
        ds_meta=dataset.meta,
        weight_pt_path="/Data/lzl/pi0-ft-simulated/pizza_task_5_lora/mp_rank_00_model_states.pt"
    )
    
    logger.info("Policy model created...")
    count_parameters_mb(policy, logger)
    logger.info("Applying the LoRA")
    lora_paligemma = policy.model.paligemma_with_expert.paligemma.merge_and_unload()
    # print(lora_paligemma)
    policy.model.paligemma_with_expert.paligemma = lora_paligemma
    torch.save(policy.state_dict(), "/Data/lzl/pi0-ft-simulated/pizza_task_5_lora/lora_merge.pt")
    print("Save")
    
    


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # https://wandb.ai/authorize
    # os.environ['WANDB_API_KEY'] = '9e1c3ac77856b8ebb5573c4e1e250c84aabfb904'
    train()