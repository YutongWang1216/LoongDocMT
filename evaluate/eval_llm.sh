#!/bin/bash

data_dir=$1
result_dir=$2
language=$3
src_lang=${language%%-*}
tgt_lang=${language##*-}

echo $result_dir

src_docs=()
tgt_docs=()
ref_docs=()
for source in $(ls $data_dir/$src_lang.* | sort -t. -k2 -n); do
    i="${source##*.}"
    src_docs+=("$data_dir/${src_lang}.$i")
    tgt_docs+=("$result_dir/${tgt_lang}.$i")
    ref_docs+=("$data_dir/${tgt_lang}.$i")
done

model="gpt-4.1"
api_key=${OPENAI_API_KEY}    # OpenAI-compatible API key
base_url=${OPENAI_BASE_URL}  # OpenAI-compatible endpoint

python -u llm.py \
    --source_files ${src_docs[@]} \
    --target_files ${tgt_docs[@]} \
    --reference_files ${ref_docs[@]} \
    --language $language \
    --output_file ${result_dir}/llm_${model}.txt \
    --model $model \
    --api_key $api_key \
    --base_url $base_url
