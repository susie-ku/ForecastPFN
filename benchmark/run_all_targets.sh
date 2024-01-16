##########################################
# Arima Autoformer FEDformer-w Informer Prophet Transformer
##########################################


##########################################
# Train Budget


# Models that require training
# is_training=1

# for model in Arima Autoformer FEDformer-w Informer Prophet Transformer 
# do

# for budget in 50 100 150 200 250 300 500 
# do

# for preLen in 6 8 14 18 24 36 48 60
# do

# targets=$(<../academic_data/exchange_rate/exchange_rate.txt)
# echo "$targets"
# for target in $targets
# do

# # exchange
# python run.py \
#  --is_training $is_training \
#  --data exchange \
#  --root_path ../academic_data/exchange_rate/ \
#  --data_path exchange_rate_.csv \
#  --model $model \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi


# targets=$(<../academic_data/weather/weather_agg.txt)
# for target in $targets
# do

# # weather
# python run.py \
#  --is_training $is_training \
#  --data weather-mean \
#  --root_path ../academic_data/weather/ \
#  --data_path weather_agg_.csv \
#  --model $model \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
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
#  --data traffic-mean \
#  --root_path ../academic_data/traffic/ \
#  --data_path traffic_agg_.csv \
#  --model $model \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
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
#  --data ECL-mean \
#  --root_path ../academic_data/electricity/ \
#  --data_path electricity_agg_.csv \
#  --model $model \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
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
#  --data ETTh1-mean \
#  --root_path ../academic_data/ETT-small/ \
#  --data_path ETTh1_agg_.csv \
#  --model $model \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
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
#  --data ETTh2-mean \
#  --root_path ../academic_data/ETT-small/ \
#  --data_path ETTh2_agg_.csv \
#  --model $model \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# done;

# done;

# done;




# ##########################################
# # Time Budget

# # time-based budget
# is_training=1

# for model in Arima Autoformer FEDformer-w Informer Prophet Transformer 
# do

# for time_budget in 1 5 10 15 30 45 60 120
# do

# for preLen in 6 8 14 18 24 36 48
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
#  --itr 5 \ \
#  --target $target

# done;

# nvidia-smi

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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
#  --itr 5 \
#  --target $target

# done;

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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --time_budget $time_budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# done;

# done;

# done;







# ##########################################
# # ForecastPFN
# ##########################################

# is_training=0
# model=ForecastPFN

# for budget in 50 #100 150 200 250 300 500  
# do

# for preLen in 6 8 14 18 24 36 48
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \ \
#  --target $target

# done;

# nvidia-smi

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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

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
#  --pred_len $preLen \
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
#  --pred_len $preLen \
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
#  --pred_len $preLen \
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
#  --pred_len $preLen \
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
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --target $target

# done;

# nvidia-smi

# done;

# done;












##########################################
# SeasonalNaive Mean Last
##########################################

# is_training=0
# for model in SeasonalNaive Mean Last 
# do

# for budget in 50 #100 150 200 250 300 500  
# do

# for preLen in 6 8 14 18 24 36 48 60
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq D \
#  --target $target

# done;

# nvidia-smi

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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq W \
#  --target $target

# done;

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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq D \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq D \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq D \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq D \
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
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --train_budget $budget \
#  --itr 5 \
#  --metalearn_freq D \
#  --target $target

# done;

# nvidia-smi


# done;

# done;

# done;

##########################################
# ForecastPFN ILI
##########################################

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

##########################################
# ForecastPFN WEATHER
##########################################

is_training=0
model=ForecastPFN

targets=$(<../academic_data/weather/weather.txt)
for target in $targets
do

# weather
python run.py \
 --is_training $is_training \
 --data weather \
 --root_path ../academic_data/weather/ \
 --data_path weather_.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len 30 \
 --train_budget 512 \
 --itr 5 \
 --target $target

done;



