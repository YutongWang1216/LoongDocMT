#!/bin/bash

in_dir=                       # parent dir containing per-language sub-dirs (e.g., en-zh/en.0, en-zh/zh.0)
out_dir=./results             # output directory for sampled trajectories
languages=(en-zh)             # one or more translation directions
urls=(127.0.0.1:8000)         # one or more deployed vLLM model APIs
comet_apis=(127.0.0.1:8088)   # one or more deployed COMET model APIs
tokenizer_path=               # path to the LLM's tokenizer
encoder_path=                 # path to the all-distilroberta-v1 checkpoint
window_size=                  # number of sentences per page within a document

python -u sample.py \
    --in_dir $in_dir \
    --out_dir $out_dir \
    --languages "${languages[@]}" \
    --urls "${urls[@]}" \
    --comet_api_list "${comet_apis[@]}" \
    --tokenizer_path $tokenizer_path \
    --encoder_path $encoder_path \
    --window_size $window_size
