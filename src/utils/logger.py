# modified from:https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/__init__.py#L12
import os 
import csv
from datetime import datetime

def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision
    
    msg = r"""
   ___           _       _   ___     __            __  ______       _    
  / _ )___ ____ (_)___  | | / (_)___/ /___ _____ _/ / / __/ /____ _(_)__ 
 / _  / _ `(_-</ / __/  | |/ / / __/ __/ // / _ `/ / _\ \/ __/ _ `/ / _ \
/____/\_,_/___/_/\__/   |___/_/_/  \__/\_,_/\_,_/_/ /___/\__/\_,_/_/_//_/
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg


# import pyfiglet
# mytext="Basic Virtual Stain"
# print(pyfiglet.figlet_format(text=mytext,font='smslant'))

def log_to_csv(name, value,ptvhs,psnr,ssim, save_path):
    """
    将数据记录到指定路径的CSV文件，最后一列为时间戳
    参数：
        name: 名称字段（字符串）
        value: 数值字段（数字）
        save_path: 完整文件路径（例如：/data/logs/records.csv）
    """
    # 准备时间戳（包含毫秒）
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留3位毫秒
    
    # 确保目录存在
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)  # 自动创建不存在的目录[5](@ref)
    
    # 处理文件头（仅在文件不存在时写入）
    header = not os.path.exists(save_path)
    
    # 写入CSV文件（追加模式）
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["Method","SSIM", "PTVH layer1" ,"PTVH layer2","PTVH layer3","PTVH layer4","FID","PSNR", "ExpTimestamp"])  # 写入表头[3](@ref)
        writer.writerow([name, ssim,ptvhs[0],ptvhs[1],ptvhs[2],ptvhs[3],value,psnr, timestamp])  # 写入数据行[1](@ref)