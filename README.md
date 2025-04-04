

Basic Virutal Stain is an open-source image and video restoration toolbox based on PyTorch, such as H&E-to-IHC,H&E-to-other virtual stain etc.

```bash
conda create -n basicvs python=3.10 -y
conda activate basicvs
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

☣ **New Features**:

☑  April 2, 2025.Add ASP training and testing codes: 

## Overview
🚀Quick start to use BasicVirtualStain or replicate our experiments in 5 minutes!

| Method                                                                                            | Dataset       | Description                                                                                                                                                                                                                | Ref Papers     | Details link                              |
|----------------------------------------------------------------------------------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|-----------------------------------------|
| [ASP]() | BCI                | Detail Setting: [configuration](./configs/BCI/ASP.yaml)                                                                                                                                                        | [MICCAI2023](https://arxiv.org/pdf/2303.06193)         | [usage]() |
| [ASP]() | MIST(HER2)                |Detail Setting: [configuration](./configs/MIST_HER2/ASP.yaml)                                                                                                                                                        | [MICCAI2023](https://arxiv.org/pdf/2303.06193)         | [usage]() |
| [ASP]() | MIST(Ki67)                    | Detail Setting: [configuration](./configs/MIST_Ki67/ASP.yaml)                                                                                                                                                        |[MICCAI2023](https://arxiv.org/pdf/2303.06193)          | [usage]() |
| [ASP]() | MIST(PR)              | Detail Setting: [configuration](./configs/MIST_PR/ASP.yaml)                                                                                                                                                        |  [MICCAI2023](https://arxiv.org/pdf/2303.06193)       | [usage]() |
| [ASP]() | MIST(ER)              | Detail Setting: [configuration](./configs/MIST_ER/ASP.yaml)                                                                                                                                                        |  [MICCAI2023](https://arxiv.org/pdf/2303.06193)       | [usage]() |



## Evaluate Metric
```bash
python scripts/metrics/calculate_ssim_folder.py 
```