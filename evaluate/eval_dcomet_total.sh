#!/bin/bash

data_dir=$1
result_dir=$2
language=$3
comet_model_path=${4:-$COMET_MODEL_PATH}  # path to wmt22-comet-da model.ckpt

echo $result_dir
src_lang=${language%%-*}
tgt_lang=${language##*-}

temp_src=$result_dir/temp_src.txt
temp_tgt=$result_dir/temp_tgt.txt
temp_ref=$result_dir/temp_ref.txt

rm $temp_src
rm $temp_tgt
rm $temp_ref
rm $result_dir/temp_doc_ids.txt

for source in $(ls $data_dir/$src_lang.* | sort -t. -k2 -n); do
    i="${source##*.}"
    source=$data_dir/$src_lang.$i
    target=$result_dir/$tgt_lang.$i
    reference=$data_dir/$tgt_lang.$i
    cat $source >> $temp_src
    cat $target >> $temp_tgt
    cat $reference >> $temp_ref

    lines=$(wc -l < ${target})
    yes $i | head -n "$lines" >> $result_dir/temp_doc_ids.txt

done
comet-score -s $temp_src -t $temp_tgt -r $temp_ref --doc $result_dir/temp_doc_ids.txt --model $comet_model_path --quiet >> $result_dir/doccomet_total.txt

cat $result_dir/doccomet_total.txt
