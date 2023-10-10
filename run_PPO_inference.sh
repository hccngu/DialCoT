startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

python -u PPO_discrete_inference.py \
--n_actions 3 \
--T 1.0 \
--device cuda:5 \
--hidden_width 1024 \
--eval_path output_ckpt/PPO_discrete_env_OurEnv-v2_number_62_seed_42 \
--eval_datasets gsm8k

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 