startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

python -u PPO_discrete_main.py \
--n_actions 3 \
--T 1.0 \
--device cuda:6 \
--number 72 \
--batch_size 4096 \
--mini_batch_size 128 \
--evaluate_freq 4096 \
--K_epochs 64 \
--lr_a 2e-4 \
--lr_c 2e-4 \
--train_dataset gsm8k \

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 