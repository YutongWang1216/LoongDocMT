
address=
language=
src_file=

src_lang=${language%-*}
tgt_lang=${language#*-}


output_dir=./results

python -u infer.py \
    --src_file $src_file \
    --output_path $output_dir \
    --window_size 10 \
    --infer_address $address \
    --schedule_address $address \
    --language ${language} \
    --infer_temperature 0.7 \
    --schedule_temperature 0.7 \
