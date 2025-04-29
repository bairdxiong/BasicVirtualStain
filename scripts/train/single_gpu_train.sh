#!/bin/bash

task="BCI"
config_name="ASP"
gpu_ids="0"
result_path="/root/Desktop/data/private/Dataset4Research/StainExp"

suffix="baseline"
exp_name="${config_name}_${suffix}"

# 解析命令行选项
while getopts "t:r:g:c:s:" opt; do
  case $opt in
    t) task="$OPTARG" ;;              # 覆盖任务名称
    r) result_path="$OPTARG" ;;       # 覆盖结果路径
    g) gpu_ids="$OPTARG" ;;           # 覆盖 GPU ID
    c) config_name="$OPTARG" ;;       # 覆盖配置名称
    s) suffix="$OPTARG" ;;            # 覆盖后缀
    *) echo "Usage: $0  [-t task] [-r result_path] [-g gpu_ids] [-c config_name] [-s suffix]" >&2
       exit 1 ;;
  esac
done

exp_name="${config_name}_${suffix}"

# 打印最终参数（可选，方便调试）
echo "Task: $task"
echo "Config Name: $config_name"
echo "GPU IDs: $gpu_ids"
echo "Result Path: $result_path"
echo "Experiment Name: $exp_name"

# 执行 Python 脚本
python -u main.py \
    --train \
    --result_path "$result_path" \
    --config "./configs/$task/$config_name.yaml" \
    --gpu_ids "$gpu_ids" \
    --exp_name "$exp_name"\
    --use_ema\
    # --use_swanlab \
    
    
    
