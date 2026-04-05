#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

EXP_NO="test"
MODALS="avl"

echo "MELD, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/MELD/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u ./code/train.py \
--dataset "MELD" \
--data_dir "./data/meld/MELD_features_raw1.pkl" \
--name ${EXP_NO} \
--modals ${MODALS} \
--log_dir ${LOG_PATH}/${EXP_NO} \
--lr 0.0005 \
--l2 0.00001 \
--gamma 0.5 \
--beta 0.5 \
--rec_dropout 0.1 \
--dropout 0.1 \
--modulation \
--tau 1 \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
