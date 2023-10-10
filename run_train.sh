#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

python -u training.py \
--output_dir output/flan-t5-large \
--use_fsdp \
--train_epochs 100 \
--max_source_length 512 \
--max_target_length 512 \
--data_path data/GSM8K/DialCoT-S-enhanced.json \
--model_name_or_path "google/flan-t5-large" \
--train_batch_size 32 \
--gradient_accumulation_steps 64

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 