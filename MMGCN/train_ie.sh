#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

EXP_NO="test"
MODALS="al"

echo "IEMOCAP, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/IEMOCAP/${MODALS}"

if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u train.py \
--dataset "IEMOCAP" \
--data_dir "./data/IEMOCAP_features/IEMOCAP_features.pkl" \
--name ${EXP_NO} \
--modals ${MODALS} \
--lr 0.0003 \
--l2 0.00003 \
--class_weight \
--log_dir ${LOG_PATH}/${EXP_NO} \
--beta 0.8 \
--gamma 0.001 \
--dropout 0.4 \
--multi_modal \
--modulation \
--tau 1 \
--use_residue \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
