
# Our Repo Vs Origin Repo

You can click url to download .pth to reproduce our repo results. The password: bsVS

Most of ori metrics comes from ASP.

部分差异来源于不同repo之间的设置，例如在ASP论文中对比的PyrmidP2P是以原论文基础的100Epoch作为评估.这里提供的都是this repo实现的权重。在论文中需要选取相同条件设置下进行对比方法。

✅ means strict obey the original repo to reproduce.

| Method                                                                                            | FID(ori/this repo)       | PSNR(ori/this repo)                                                                                                                                                                                                                | SSIM(ori/this repo)    | Dataset                               |
|----------------------------------------------------------------------------------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|-----------------------------------------|
| ASP | 50.13/59.99                |    14.662/14.427                                                                                                                                            |  0.1988/0.xx       | [HER2(Cfg)](../../configs/MIST_HER2/ASP.yaml) 20epoch | 
| ✅PyrmidPix2Pix | -/92.61                |    21.160/20.497                                                                                                                                            |  0.477/0.486       | [BCI(Cfg)](../../configs/BCI/PyrmidP2P.yaml)  [100epoch](https://pan.baidu.com/s/116efR1qBHNBW_2JnGT4DOw?pwd=bsVS) | 
| ASP | 41.027/75.539                |    13.9554/14.1819                                                                                                                                            |  0.2045/0.2123      | [ER(Cfg)](../../configs/MIST_HER2/ASP.yaml) [40epoch](https://pan.baidu.com/s/1khIVq4UbUxCf0bB3O9KS7A?pwd=bsVs) | 
| ✅ASP | 41.027/66.675                |    13.9554/14.3372                                                                                                                                            |  0.2045/0.2231      | [ER(Cfg)](../../configs/MIST_HER2/ASP.yaml) [30epoch](https://pan.baidu.com/s/1fLUulPskZKW7i31ni4dwBw?pwd=bsVS) | 
| PyrmidPix2Pix | 107.4/157.926                |    -/13.876                                                                                                                                            |  0.2172/0.2104       | [ER(Cfg)](../../configs/MIST_ER/PyrmidP2P.yaml) [40epoch](https://pan.baidu.com/s/1e1NxRbo9nMzAnSKzxv_95Q?pwd=bsVS) | 
| ✅PyrmidPix2Pix | 107.4/120.69                |    -/14.56                                                                                                                                            |  0.2172/0.2512       | [ER(Cfg)](../../configs/MIST_ER/PyrmidP2P.yaml) [100epoch](https://pan.baidu.com/s/1ZXgoXuc3FuweHhNk1OQVbQ?pwd=bsVS) | 


# Ablation For MIST(ER) dataset

The following table is the same params setting.

| Method                                                                                            | FID      | PSNR                                                                                                                                                                                                                | SSIM   | Dataset                               |
|----------------------------------------------------------------------------------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|-----------------------------------------|
| CUT | 63.808                |    13.3070                                                                                                                                            |  0.1967      | [ER(Cfg)](../../configs/MIST_ER/CUT.yaml) [40epoch](https://pan.baidu.com/s/1dx-X9O80KKO7FcLjoX4f8w?pwd=bsVS) | 
| PyrmidPix2Pix | 157.926                |    13.876                                                                                                                                            |  0.2104       | [ER(Cfg)](../../configs/MIST_ER/PyrmidP2P.yaml) [40epoch](https://pan.baidu.com/s/1e1NxRbo9nMzAnSKzxv_95Q?pwd=bsVS) | 
| ASP | 75.539                |    14.1819                                                                                                                                            |  0.2123      | [ER(Cfg)](../../configs/MIST_HER2/ASP.yaml) [40epoch](https://pan.baidu.com/s/1khIVq4UbUxCf0bB3O9KS7A?pwd=bsVs) | 
| TDKStain | x               |    x                                                                                                                                            |  x      | [ER(Cfg)](../../configs/MIST_HER2/ASP.yaml) [40epoch]() | 
| PPT | x               |    x                                                                                                                                            |  x      | [ER(Cfg)](../../configs/MIST_HER2/ASP.yaml) [40epoch]() | 
| PSPStain | x               |    x                                                                                                                                            |  x      | [ER(Cfg)](../../configs/MIST_HER2/ASP.yaml) [40epoch]() | 

## Train your own dataset
```bash
bash scripts/train/single_gpu_train.sh -t xxx(dataset_type) -c xxx(config_name)  -s xxxx(exp_name) -g x(gpu_ids)
```
dataset_type will decide you chose which folder in configs. such as MIST_ER, and config_name will chose this folder which yaml. such as -t MIST_ER -s PyrmidP2P, it means : configs/MIST_ER/PyrmidP2P.yaml