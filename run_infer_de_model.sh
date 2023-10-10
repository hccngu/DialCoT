#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

model_type_lst=(Flan-T5-XL)

test_dataset_lst=(gsm8k)
for model_type in "${model_type_lst[@]}"; do
    for test_dataset in "${test_dataset_lst[@]}"; do
        python -u inference_declare_model.py --test_dataset $test_dataset --model_type $model_type
    done
done
endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 