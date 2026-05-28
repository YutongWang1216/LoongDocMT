#!/bin/bash

data_dir=$1
result_dir=$2
language=$3
url=${4:-127.0.0.1:8090}

echo $result_dir
python scomet.py -d $data_dir -r $result_dir -l $language -u $url
