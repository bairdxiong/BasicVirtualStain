
EXP_NAME="ASP"
DATASET_TYPE="MIST_HER2"
DATASETROOT="/root/Desktop/data/private/Dataset4Research/StainExp/BCI_dataset/ASP_baseline/image"

python scripts/metrics/eval_metric.py  --dataroot $DATASETROOT  --dataset_type $DATASET_TYPE --exp_name $EXP_NAME