#!/bin/bash

cd "$(dirname "$0")"

model_path=         # path to the COMET checkpoint (e.g., wmt22-comet-da/model.ckpt)
port=8090           # listening port

CUDA_VISIBLE_DEVICES=0 COMET_GPUS=1 python -u deploy.py --model_path $model_path --port $port &> deploy.log &
echo "Started, pid=$!, logs at deploy.log"
