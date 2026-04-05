#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

EXP_NO="test_vl2"
MODALS="vl"

echo "IEMOCAP, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/IEMOCAP/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u ./code/train.py \
--name ${EXP_NO} \
--modals ${MODALS} \
--dataset "IEMOCAP" \
--data_dir "./data/iemocap/IEMOCAP_features.pkl" \
--log_dir ${LOG_PATH}/${EXP_NO} \
--lr 0.0001 \
--l2 0.00001 \
--class_weight \
--gamma 0.5 \
--beta 0.5 \
--rec_dropout 0.1 \
--dropout 0.1 \
--modulation \
--tau 1 \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1