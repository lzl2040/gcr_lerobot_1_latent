## Run on Local with Single Dataset
you need modify:
- ```/home/v-wangxiaofa/lzl/gcr_lerobot_1/lerobot/configs/policies.py```: 45 line, ```pretrained_path```
- ```/home/v-wangxiaofa/lzl/gcr_lerobot_1/lerobot/common/policies/pi0/modeling_pi0.py```: 251 line, ```self.language_tokenizer```

## Run on Physics Inteligence
- ```/home/v-wangxiaofa/lzl/gcr_lerobot_1/lerobot/common/policies/factory.py```: 125 line
- ```/home/v-wangxiaofa/lzl/gcr_lerobot_1/lerobot/common/constants.py```: ```STATE``` and ```ACTION```  
- ```/home/v-wangxiaofa/lzl/gcr_lerobot_1/lerobot/common/datasets/factory.py```: 59 line