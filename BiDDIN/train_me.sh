#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

EXP_NO="test"
MODALS="av"

echo "MELD, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/MELD/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u train_bimodal.py \
--dataset "MELD" \
--data_dir "./data/MELD_features/MELD_features_raw1.pkl" \
--name ${EXP_NO} \
--modals ${MODALS} \
--tensorboard \
--beta 0.8 \
--gamma 0.05 \
--lr 0.0001 \
--modulation \
--dropout 0.1 \
--log_dir ${LOG_PATH}/${EXP_NO} \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
