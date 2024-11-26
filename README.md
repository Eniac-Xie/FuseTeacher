# FuseTeacher
## Dataset
* The YFCC15M-Cap dataset has been released: [YFCC15M-Cap](https://huggingface.co/datasets/xiaociwei/YFCC15M-LLaVA-Cap)
* The Union23M and Union65M datasets are coming soon.

## Training
Train FuseTeacher on YFCC15M-Cap dataset:
```
sh train.sh
```

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{FuseTeacher2024,
  title = {FuseTeacher: Modality-fused Encoders are Strong Vision Supervisors},
  author = {Xie, Chen-Wei and Sun, Siyang and Zhao, Liming and Li, Pandeng and Ma, Shuailei and Zheng, Yun},
  booktitle = {ECCV},
  year = {2024}
}
```

Some code is borrowed from [ALBEF](https://github.com/salesforce/ALBEF), [CLIP](https://github.com/openai/CLIP), and [timm](https://github.com/huggingface/pytorch-image-models). Thanks a lot to them.
