#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

EXP_NO="test_avl1"
MODALS="avl"

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
--nodal-attention \
--lr 0.0003 \
--l2 0.0 \
--class_weight \
--gamma 0.05 \
--beta 0.0 \
--dropout 0.4 \
--modulation \
--tau 1 \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
