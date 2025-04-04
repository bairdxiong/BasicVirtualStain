

---

🧬 BasicVS (**Basic** **V**irtual **S**tain) is an open-source image and video restoration toolbox based on PyTorch, such as H&E-to-IHC,H&E-to-other virtual stain etc.

##  Based Environment 
```bash
conda create -n basicvs python=3.10 -y
conda activate basicvs
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

☣ **New Features**:

✅  April 2, 2025.Add Adaptive Supervised PatchNCE(MICCAI2023) training and testing codes: 

If BasicVS helps your research or work, please help to ⭐ this repo or recommend it to your friends. Thanks😊 <br>

## Overview
🚀Quick start to use BasicVirtualStain or replicate our experiments in 5 minutes!

| Method                                                                                            | Dataset       | Description                                                                                                                                                                                                                | Ref Papers     | Details link                              |
|----------------------------------------------------------------------------------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|-----------------------------------------|
| [ASP]() | BCI/MIST                | Detail Setting: [configuration](./configs/BCI/ASP.yaml)                                                                                                                                                        | [MICCAI2023](https://arxiv.org/pdf/2303.06193)         | [usage]() |




## Evaluate Metric
```bash
python scripts/metrics/calculate_ssim_folder.py 
```


## Acknowledgement
Our code was inspired based on the code from [BBDM](https://github.com/xuekt98/BBDM). We are grateful to Bo Li,Kai-Tao Xue, et al.