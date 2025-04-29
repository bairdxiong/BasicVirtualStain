
task="MIST_ER"
config_name="SIMGAN"
gpu_ids="0"
batch=1
result_path="/root/Desktop/data/private/Dataset4Research/StainExp"

suffix="baseline"
exp_name="${config_name}_${suffix}"

python -u main.py\
    --result_path $result_path\
    --config ./configs/$task/$config_name.yaml\
    --gpu_ids $gpu_ids\
    --exp_name $exp_name\

python scripts/metrics/evaluate_score.sh -e SIMGAN