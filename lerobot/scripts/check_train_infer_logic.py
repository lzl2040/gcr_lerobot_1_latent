from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.datasets.lerobot_dataset import MultiSameDataset, MultiDatasetforDistTraining
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from torch.utils.data import DataLoader
from lerobot.common.datasets.transforms import ImageTransforms
import torch
import numpy as np

def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@parser.wrap()
def check_train(cfg: TrainPipelineConfig):
    cfg.validate()
    set_seed(1001)
    
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    dataset = MultiSameDataset(cfg=cfg, 
                               image_transforms=image_transforms,
                            #    wrist_image_transforms=wrist_image_transforms)
    )
    # print(dataset.stats["action"]["mean"][:14], dataset.stats["action"]["std"][:14])
    weight_path = "/Data/lzl/pi0-ft-simulated/pizza_task_12_and_2/mp_rank_00_model_states.pt"
    policy = make_policy(
        cfg=cfg.policy,
        device='cpu',
        ds_meta=dataset.meta,
        weight_pt_path=weight_path
    )
    
    device = "cuda:0"
    for params in policy.parameters():
        params.data = params.data.bfloat16()
    policy = policy.to(device=device)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1)
    
    state_mean = torch.from_numpy(dataset.stats["observation.state"]["mean"][:15]).to(dtype=torch.bfloat16).to(device)
    state_std = torch.from_numpy(dataset.stats["observation.state"]["std"][:15]).to(dtype=torch.bfloat16).to(device)
    
    # print(f"State Mean: {state_mean}, State Std: {state_std}")
    # State Mean: tensor([ 0.4609, -0.1299,  0.2695, -0.0317,  0.9961,  0.0243,  0.0284,  0.0222,
    #     -0.1406, -0.1387, -0.1445, -2.3594,  0.0938,  2.2656,  0.3906],
    #    device='cuda:0', dtype=torch.bfloat16), State Std: tensor([0.0513, 0.0669, 0.0649, 0.0337, 0.0033, 0.0520, 0.0408, 0.0325, 0.3105,
    #     0.2734, 0.3223, 0.1738, 0.1318, 0.1611, 0.1719], device='cuda:0',
    #    dtype=torch.bfloat16)
    
    action_mean = dataset.stats["action"]["mean"][:14]
    action_std = dataset.stats["action"]["std"][:14]
    # print(f"Action Mean: {action_mean}, Action Std: {action_std}")
    #     Action Mean: [ 1.09537981e-04 -3.92437743e-04 -2.24346681e-04 -1.44424903e-04
    #  -5.28488396e-04  2.88175764e-04 -5.57814639e-01 -6.82646086e-04
    #   8.25115996e-04 -1.68939324e-04  1.67407168e-04  1.55017986e-04
    #   4.06611469e-05 -7.98381565e-04], Action Std: [0.00598355 0.00660015 0.00758196 0.0124057  0.01137537 0.01371028
    #  0.82996561 0.01270723 0.02964769 0.01359949 0.01924217 0.01848486
    #  0.01949782 0.02262247]
    noise = torch.ones(1, 50, 32).to(dtype=torch.bfloat16).to(device)
    # imgs = torch.ones(1, 3, 224, 224).to(dtype=torch.bfloat16).to(device) # 1 3 224 224
    # secondary_imgs = torch.ones(1, 3, 224, 224).to(dtype=torch.bfloat16).to(device) # 1 3 224 224
    # wrist_imgs = torch.ones(1, 3, 224, 224).to(dtype=torch.bfloat16).to(device) # 1 3 224 224
    # task = ["pick coke can"]
    # state = torch.ones(1, 15).to(dtype=torch.bfloat16).to(device) # 1 15
    # batch = {
    #     "task": task,
    #     "observation.state": state,
    #     "observation.images.primary": imgs,
    #     "observation.images.secondary": secondary_imgs,
    #     "observation.images.wrist": wrist_imgs,  
    # }
    # actions = policy.infer(batch, noise=noise)
    # print(actions[0, 0])
    # Actions after unnorm: tensor([ 1.7776e-03,  4.9744e-03,  3.2654e-03,  2.0142e-02,  1.3367e-02,
    #      3.7994e-03,  9.9609e-01, -5.9128e-04,  1.3123e-03, -3.3569e-04,
    #     -5.8174e-05,  8.3160e-04,  4.0770e-05, -1.4191e-03], device='cuda:0',
    #    dtype=torch.bfloat16)
    for batch in dataloader:
        imgs = batch["observation.images.primary"].to(dtype=torch.bfloat16).to(device)
        secondary_imgs = batch["observation.images.secondary"].to(dtype=torch.bfloat16).to(device)
        wrist_imgs = batch["observation.images.wrist"].to(dtype=torch.bfloat16).to(device)
        task = batch["task"]
        state = batch["observation.state"].to(dtype=torch.bfloat16).to(device)
        
        state_norm = (state - state_mean) / (state_std + 1e-8)
        # print(f"State Norm: {state_norm}")
        # State Norm: tensor([[ 0.4609, -0.1299,  0.2695, -0.0317,  0.9961,  0.0243,  0.0284,
        #         0.0222, -0.1406, -0.1387, -0.1445, -2.3594,  0.0938,  2.2656,
        #         0.3906]], device='cuda:0', dtype=torch.bfloat16)
        
        batch = {
            "task": task,
            "observation.state": state,
            "observation.images.primary": imgs,
            "observation.images.secondary": secondary_imgs,
            "observation.images.wrist": wrist_imgs,
        }
        
        actions = policy.infer(batch, noise=noise)
        print(actions[0, 0])
    #     tensor([-6.4850e-04,  6.5994e-04, -5.9509e-04,  2.2430e-03, -3.6316e-03,
    #     -7.5378e-03, -9.9219e-01, -6.4087e-04,  9.6893e-04, -2.5558e-04,
    #      2.1553e-04,  1.7357e-04,  8.8692e-05, -8.6212e-04], device='cuda:0',
    #    dtype=torch.bfloat16)
        break

@parser.wrap()
def check_infer(cfg: TrainPipelineConfig):
    cfg.validate()
    set_seed(1001)
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    dataset = MultiSameDataset(cfg=cfg, 
                               image_transforms=image_transforms,
                            #    wrist_image_transforms=wrist_image_transforms)
    )
    dataset_multi = MultiDatasetforDistTraining(cfg=cfg,
                                          data_mix=cfg.dataset.data_mix,
                                          vla2root_json="/home/v-wangxiaofa/lzl/gcr_lerobot_1/vla2root.json",
                                          image_transforms=image_transforms)
    
    weight_path = "/Data/lzl/pi0-ft-simulated/pizza_task_12_and_2/mp_rank_00_model_states.pt"
    policy = make_policy(
        cfg=cfg.policy,
        device='cpu',
        ds_meta=dataset.meta,
        weight_pt_path=weight_path
    )
    
    device = "cuda:0"
    for params in policy.parameters():
        params.data = params.data.bfloat16()
    policy = policy.to(device=device)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1)
    
    state_mean_multi = dataset_multi.stats["observation.state"]["mean"][:15].to(dtype=torch.bfloat16).to(device)
    state_std_multi = dataset_multi.stats["observation.state"]["std"][:15].to(dtype=torch.bfloat16).to(device)
    # print(f"State Mean: {state_mean_multi}, State Std: {state_std_multi}")
    # State Mean: tensor([ 0.4609, -0.1299,  0.2695, -0.0317,  0.9961,  0.0243,  0.0284,  0.0222,
    #     -0.1406, -0.1387, -0.1445, -2.3594,  0.0938,  2.2656,  0.3906],
    #    device='cuda:0', dtype=torch.bfloat16), State Std: tensor([0.0513, 0.0669, 0.0649, 0.0337, 0.0033, 0.0520, 0.0408, 0.0325, 0.3105,
    #     0.2734, 0.3223, 0.1738, 0.1318, 0.1611, 0.1719], device='cuda:0',
    #    dtype=torch.bfloat16)
    # action
    action_mean_multi = dataset_multi.stats["action"]["mean"][:14]
    action_std_multi = dataset_multi.stats["action"]["std"][:14]
    # print(f"Action Mean: {action_mean_multi}, Action Std: {action_std_multi}")
    # Action Mean: tensor([ 1.0954e-04, -3.9244e-04, -2.2435e-04, -1.4442e-04, -5.2849e-04,
    #      2.8818e-04, -5.5781e-01, -6.8265e-04,  8.2512e-04, -1.6894e-04,
    #      1.6741e-04,  1.5502e-04,  4.0661e-05, -7.9838e-04],
    #    dtype=torch.float64), Action Std: tensor([0.0060, 0.0066, 0.0076, 0.0124, 0.0114, 0.0137, 0.8300, 0.0127, 0.0296,
    #     0.0136, 0.0192, 0.0185, 0.0195, 0.0226], dtype=torch.float64)
    noise = torch.ones(1, 50, 32).to(dtype=torch.bfloat16).to(device) # 1 50 32
    # imgs = torch.ones(1, 3, 224, 224).to(dtype=torch.bfloat16).to(device) # 1 3 224 224
    # secondary_imgs = torch.ones(1, 3, 224, 224).to(dtype=torch.bfloat16).to(device) # 1 3 224 224
    # wrist_imgs = torch.ones(1, 3, 224, 224).to(dtype=torch.bfloat16).to(device) # 1 3 224 224
    # task = ["pick coke can"]
    # state = torch.ones(1, 15).to(dtype=torch.bfloat16).to(device) # 1 15
    # state_norm = (state - state_mean_multi) / (state_std_multi + 1e-8)
    # batch = {
    #     "task": task,
    #     "observation.state": state_norm,
    #     "observation.images.primary": imgs,
    #     "observation.images.secondary": secondary_imgs,
    #     "observation.images.wrist": wrist_imgs,
    # }
    # actions = policy.infer_wo_norm(batch, noise=noise).to(dtype=torch.float32).detach().cpu() # 1 50 32
    # actions = (actions * (action_std_multi + 1e-8)) + action_mean_multi
    # actions = actions.numpy()
    # print("Actions after unnorm:", actions[0, 0])
    #     Actions after unnorm: [ 1.78072671e-03  4.97018873e-03  3.27046577e-03  2.01117748e-02
    #   1.34241310e-02  3.79608322e-03  9.98370898e-01 -5.89575459e-04
    #   1.31731413e-03 -3.34948880e-04 -5.80870862e-05  8.31954271e-04
    #   4.06611469e-05 -1.41696500e-03]
    for batch in dataloader:
        imgs = batch["observation.images.primary"].to(dtype=torch.bfloat16).to(device)
        secondary_imgs = batch["observation.images.secondary"].to(dtype=torch.bfloat16).to(device)
        wrist_imgs = batch["observation.images.wrist"].to(dtype=torch.bfloat16).to(device)
        task = batch["task"]
        state = batch["observation.state"].to(dtype=torch.bfloat16).to(device)
        
        # Normalize state
        state_norm = (state - state_mean_multi) / (state_std_multi + 1e-8)
        batch = {
            "task": task,
            "observation.state": state_norm,
            "observation.images.primary": imgs,
            "observation.images.secondary": secondary_imgs,
            "observation.images.wrist": wrist_imgs,
        }
        actions = policy.infer_wo_norm(batch, noise=noise).to(dtype=torch.float32).detach().cpu() # 1 50 32
        actions = (actions * (action_std_multi + 1e-8)) + action_mean_multi
        actions = actions.numpy()
        print("Actions after unnorm:", actions[0, 0])
        #         Actions after unnorm: [-6.50093260e-04  6.58172973e-04 -5.94559864e-04  2.24222064e-03
        #  -3.63894495e-03 -7.53097673e-03 -9.92249769e-01 -6.39213127e-04
        #   9.69880154e-04 -2.55264293e-04  2.14385137e-04  1.73069620e-04
        #   8.82632787e-05 -8.64658361e-04]
        break
    

if __name__ == "__main__":
    check_infer()
    # check_train()