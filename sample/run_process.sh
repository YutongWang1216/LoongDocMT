for language in en-zh en-de en-fr; do

    echo "################## sft-$language ##################"

    python -u process.py \
        --input_path  \
        --output_path ./ \
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
        --input_path  \
        --output_path ./ \
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
