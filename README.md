## Dependency

Create a new conda virtual environment:
```
conda create -n LGASSNet python==3.8.5
conda activate LGASSNet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
pip install timm numpy opencv-python scipy matplotlib einops antialiased_cnns tqdm pandas optuna
```

## Pretrained Weights of LGASSNet
Pre-trained models for lettuce phenotypic trait analysis:

Lettuce 400-epoch pre-trained LGASSNet_l0_weights: [Download](https://github.com/qiuguangjie87/PP_Phenotypic_Fingerprint/raw/main/PP_Phenotypic_Fingerprint_main/LGASSNet_Lettuce_main/weights/LGASSNet_l0_weights.pth)

Lettuce 400-epoch pre-trained LGASSNet_l2_weights: [Download](https://github.com/qiuguangjie87/PP_Phenotypic_Fingerprint/raw/main/PP_Phenotypic_Fingerprint_main/LGASSNet_Lettuce_main/weights/LGASSNet_l2_weights.pth)



## Pretrained Weights of Backbones
If you want to train from scratch, download the following ImageNet pre-trained backbones:


Imagenet 300-epoch pre-trained LWGANet-L0 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l0_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L1 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l1_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L2 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l2_e296.pth)



## References

Backbone:
@inproceedings{lu2026lwganet,
  title={LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention},
  author={Lu, Wei and Yang, Xue and Chen, Si-Bao},
  booktitle={AAAI Conference on Artificial Intelligence},
  pages={},
  year={2026}
}

EMA:
@inproceedings{ouyang2023ema,
  title={Efficient Multi-Scale Attention Module with Cross-Spatial Learning},
  author={Ouyang, Daliang and He, Su and Zhan, Jian and Guo, Huaiyong and Huang, Zhijie and Luo, Mingzhu and Zhang, Guozhong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1-5},
  year={2023}
}
