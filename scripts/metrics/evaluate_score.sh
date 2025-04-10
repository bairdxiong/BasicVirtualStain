#!/bin/bash

# 默认值
EXP_NAME="ASP"
DATASET_TYPE="MIST_ER"
DATASETROOT="/root/Desktop/data/private/Dataset4Research/StainExp"

# 解析命令行选项
while getopts "e:t:r:" opt; do
  case $opt in
    e) EXP_NAME="$OPTARG" ;;
    t) DATASET_TYPE="$OPTARG" ;;
    r) DATASETROOT="$OPTARG" ;;
    *) echo "Usage: $0 [-e exp_name] [-t dataset_type] [-r datasetroot]" >&2
       exit 1 ;;
  esac
done

# 执行 Python 脚本
python scripts/metrics/eval_metric.py \
  --dataroot "$DATASETROOT" \
  --dataset_type "$DATASET_TYPE" \
  --exp_name "$EXP_NAME"