input_path=         # directory holding per-chapter trajectories emitted by run_sample.sh
output_path=./      # directory to write SFT/DPO json files and dataset_info.json
tokenizer_path=     # path to the LLM's tokenizer (used for length filtering)

for language in en-zh en-de en-fr zh-en de-en fr-en; do

    echo "################## sft-$language ##################"

    python -u process.py \
        --input_path $input_path \
        --output_path $output_path \
        --tokenizer_path $tokenizer_path \
        --max_length 2560 \
        --balanced True \
        --na_ratio 0.33 \
        --trans_tool_ratio 1.5 \
        --stage sft \
        --format openai \
        --multi_pairs False \
        --trans_style base \
        --trans_label sample \
        --merge_data False \
        --language $language \
        --max_docs 500

    echo "################## dpo-$language ##################"

    python -u process.py \
        --input_path $input_path \
        --output_path $output_path \
        --tokenizer_path $tokenizer_path \
        --max_length 2560 \
        --balanced True \
        --na_ratio 0.33 \
        --trans_tool_ratio 1.5 \
        --stage dpo \
        --format sharegpt \
        --multi_pairs False \
        --trans_style base \
        --trans_label sample \
        --merge_data False \
        --language $language \
        --max_docs 500
done
