
task="BCI"
config_name="PyrmidP2P"
gpu_ids="1"
result_path="/root/Desktop/data/private/Dataset4Research/StainExp"

suffix="baseline"
exp_name="${config_name}_${suffix}"

python -u main.py\
    --train \
    --result_path $result_path\
    --config ./configs/$task/$config_name.yaml\
    --gpu_ids $gpu_ids\
    --exp_name $exp_name\