

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
| [ASP]() | BCI/MIST                | Detail Setting: [configuration](./configs/BCI/ASP.yaml)                                                                                                                                                        | [MICCAI2023](https://arxiv.org/pdf/2303.06193)         | [usage](./scripts/train/asp_train_bci.sh) |
| [PyrmidPix2Pix]() | BCI/MIST                | Detail Setting: [configuration](./configs/BCI/PyrmidP2P.yaml)                                                                                                                                                        | [CVPR2022](https://arxiv.org/pdf/2204.11425v1)         | [usage](./scripts/train/pyrmidp2p_train_bci.sh) |

## Quick Training
```
bash scripts/train/xxx.sh(your .sh file)
```
More details you can read [here](./assets/TrainREADME.md)

## Evaluate Metric
```bash
python scripts/metrics/calculate_ssim_folder.py 
```


## Acknowledgement
Our code was inspired based on the code from [BBDM](https://github.com/xuekt98/BBDM). We are grateful to Bo Li,Kai-Tao Xue, et al.

📜 License and Acknowledgement
This project is released under the Apache 2.0 license.

More details about license and acknowledgement are in [LICENSE](./LICENSE/LICENSE).


## 🌏 Citations

If BasicSR helps your research or work, please cite BasicVS.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{basicvs,
  author =       {Bing Xiong},
  title =        {{BasicVS}: Open Source Histopathology Virtual Stain Toolbox},
  howpublished = {\url{https://github.com/bairdxiong/BasicVirtualStain}},
  year =         {2025}
}
```



## 📧 Contact

If you have any questions, please email `b.xiong@siat.ac.cn`.

<br>