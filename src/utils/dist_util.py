"""
modified from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/dist_util.py and https://github.com/FoundationVision/LlamaGen/blob/main/utils/distributed.py
Helpers for distributed training.
"""

import io
import os
import socket
import torch
import datetime
import subprocess
import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist



def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")

def init_distributed_mode(args):
    # Torchrun 兼容性处理
    if 'LOCAL_RANK' in os.environ:  # Torchrun 会自动设置该变量
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    
    # 新增 NCCL 超时设置（PyTorch 1.13+ 要求）
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 设置1秒超时
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
    
    # 修改后端初始化逻辑
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    
    # 使用新版初始化API
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=30)  # 显式设置超时
    )
    
    # 新增屏障同步和验证
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"成功初始化分布式训练组，共 {args.world_size} 个进程")
        
    # 设置 cuda 工作流 (PyTorch 1.13+ 优化项)
    torch.cuda.set_per_process_memory_fraction(0.5)  # 限制显存占用

