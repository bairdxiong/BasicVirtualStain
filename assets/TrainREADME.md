
# How to train your own data

## describe params in shell

For example:
```bash
task="BCI"  # this is choose which dataset, it will impact which config folder to read.
config_name="ASP" # read yaml config from: ./config/task/config_name.yaml
gpu_ids="0"
result_path="/root/Desktop/data/private/Dataset4Research/StainExp" # this is where your experiment to save

suffix="baseline" # your experiment name to save
exp_name="${config_name}_${suffix}"

python -u main.py\
    --train \  # if you want to test,just remove it
    --result_path $result_path\
    --config ./configs/$task/$config_name.yaml\
    --gpu_ids $gpu_ids\
    --exp_name $exp_name\
```