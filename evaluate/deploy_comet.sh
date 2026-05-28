#!/bin/bash

cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=4 COMET_GPUS=1 python -u deploy.py --port 8090 &> deploy.log &
echo "Started, pid=$!, logs at deploy.log"
