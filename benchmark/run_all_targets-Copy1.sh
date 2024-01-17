##########################################
# ForecastPFN
##########################################

is_training=0
model=ForecastPFN

# for seqLen in 48 96 192 384 490
# do

# targets=$(<../academic_data/exchange_rate/exchange_rate.txt)
# for target in $targets
# do


# # exchange
# python run.py \
#  --is_training $is_training \
#  --data exchange \
#  --root_path ../academic_data/exchange_rate/ \
#  --data_path exchange_rate_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len $seqLen \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget 1 \
#  --itr 5 \
#  --target $target

# done;

# done;

for seqLen in 24 48 96 192 384 768 1536 3072
do

targets=$(<../academic_data/illness/national_illness.txt)
for target in $targets
do

# illness
python run.py \
 --is_training $is_training \
 --data ili \
 --root_path ../academic_data/illness/ \
 --data_path national_illness_.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len $seqLen \
 --label_len 18 \
 --pred_len 24 \
 --train_budget 1 \
 --itr 5 \
 --target $target

done;

done;

# nvidia-smi

# targets=$(<../academic_data/weather/weather_agg.txt)
# for target in $targets
# do

# # weather
# python run.py \
#  --is_training $is_training \
#  --data weather \
#  --root_path ../academic_data/weather/ \
#  --data_path weather_agg_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# targets=$(<../academic_data/traffic/traffic_agg.txt)
# for target in $targets
# do


# # traffic
# python run.py \
#  --is_training $is_training \
#  --data traffic \
#  --root_path ../academic_data/traffic/ \
#  --data_path traffic_agg_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# targets=$(<../academic_data/electricity/electricity_agg.txt)
# for target in $targets
# do


# # electricity
# python run.py \
#  --is_training $is_training \
#  --data ECL \
#  --root_path ../academic_data/electricity/ \
#  --data_path electricity_agg_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# targets=$(<../academic_data/ETT-small/ETTh1_agg.txt)
# for target in $targets
# do


# # ETTh1
# python run.py \
#  --is_training $is_training \
#  --data ETTh1 \
#  --root_path ../academic_data/ETT-small/ \
#  --data_path ETTh1_agg_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# targets=$(<../academic_data/ETT-small/ETTh2_agg.txt)
# for target in $targets
# do


# # ETTh2
# python run.py \
#  --is_training $is_training \
#  --data ETTh2 \
#  --root_path ../academic_data/ETT-small/ \
#  --data_path ETTh2_agg_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# done;


# #########################################
# # ForecastPFN ILI
# #########################################

# is_training=0
# model=ForecastPFN

# targets=$(<../academic_data/illness/national_illness.txt)
# for target in $targets
# do

# # illness
# python run.py \
#  --is_training $is_training \
#  --data ili \
#  --root_path ../academic_data/illness/ \
#  --data_path national_illness_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 148 \
#  --label_len 18 \
#  --pred_len 24 \
#  --train_budget 1 \
#  --itr 5 \
#  --target $target

# done;

# done;

# ##########################################
# # ForecastPFN WEATHER
# ##########################################

# is_training=0
# model=ForecastPFN

# targets=$(<../academic_data/weather/weather.txt)
# for target in $targets
# do

# # weather
# python run.py \
#  --is_training $is_training \
#  --data weather \
#  --root_path ../academic_data/weather/ \
#  --data_path weather_.csv \
#  --model $model \
#  --model_path ../saved_weights \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len 96 \
#  --train_budget 1 \
#  --itr 5 \
#  --target $target

# done;



