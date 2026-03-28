#!/bin/bash
TRAIN_SCRIPT="./scripts/stable/sdxl_train.py"
accelerate launch --num_processes=1 "$TRAIN_SCRIPT" --dataset_config="dataset1.toml" --config_file="config1.toml"    --use_zero_cond=False --ddp_timeout 3600 --skip_existing --save_state --save_state_on_train_end --flow_model  --flow_use_ot  --flow_timestep_distribution uniform --flow_uniform_static_ratio 2 --skip_existing --prefetch_factor 2 
echo "All training jobs finished. Press any key to close..." 
read -n 1 -s -r -p ""