### HC-MAE: Hierarchical Cross-attention Masked Autoencoder Integrating Histopathological Images and Multi-omics for Cancer Survival Prediction

This is a PyTorch implementation of the [HC-MAE](https://wi-lab.com/cyberchair/2023/bibm23/yourpaper/B396_8612.pdf) under Linux with GPU NVIDIA A100 80GB.

### Requirements
- pytorch 1.8.0+cu111
- Pillow 9.5.0
- timm 0.3.2
- lifelines 0.27.4

### Download
- [MAE-Patch checkpoint]().
- [MAE-Region checkpoint]().
- [HC-MAE checkpoint]().


### Run
Download the whole slide images first, and then move the images to the path './dataset/'. 

1. Pre-training MAE-Patch
```angular2htm
  cd ./pretrain_stage1
  CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py
  CUDA_VISIBLE_DEVICES=0 python3 main_embedding.py
```

2. Pre-training MAE-Region
```angular2htm
  cd ./pretrain_stage2
  CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py
  CUDA_VISIBLE_DEVICES=0 python3 main_embedding.py
```

3. Survival prediction
```angular2html
  cd ./survival
  CUDA_VISIBLE_DEVICES=0 python3 main_finetune_hc_mae.py
```
