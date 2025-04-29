"""
Author: Bing Xiong Date: 2025-3-31
"""
import os 
import torch
import copy
import random
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from dataset import get_dataset
from Runner import get_runner
from src.utils.logger import get_env_info
from src.utils.args_util import dict2namespace,parse_from_yaml,namespace2dict

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    parser.add_argument('-c', '--config', type=str, default='./configs/BCI/ASP.yaml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=24, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")
    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    # system
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')
    parser.add_argument('--exp_name', type=str, help='experiment name (result dir name)')
    parser.add_argument('--use_ema', action='store_true',help="if use ema or not")
    parser.add_argument('--use_swanlab', action='store_true',help="if use swanlab or not")
    parser.add_argument('--swanlab_mode',  type=str, default='cloud',help="swanlab mode: local or cloud")
    # diffusion part
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")
    
    args = parser.parse_args()
    # with open(args.config, 'r') as f:
    dict_config = parse_from_yaml(args.config)
     
    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args 
    dict_config = namespace2dict(namespace_config)

    return namespace_config

def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def DDP_launcher(world_size, run_fn, config):
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)

def DDP_run_fn(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    set_random_seed(config.args.seed)
    assert config.data.train.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    config.training.device = [torch.device("cuda:%d" % local_rank)]
    print('using device:', config.training.device)
    config.training.local_rank = local_rank
    print(f"create {config.runner}...")
    runner = get_runner(config.runner, config) # get registered with name runner
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return

def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def main():
    nconfig = parse_args_and_config()
    args = nconfig.args 
    msg = get_env_info()
    print(msg)
    gpu_ids = args.gpu_ids
    if gpu_ids == "-1":
        nconfig.training.use_DDP=False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if len(gpu_list)>1:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)
    return 

if __name__ == "__main__":
    main()