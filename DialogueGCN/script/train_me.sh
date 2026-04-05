#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

EXP_NO="test"
MODALS="avl"

echo "MELD, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/MELD/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u ./code/train.py \
--name ${EXP_NO} \
--modals ${MODALS} \
--dataset "MELD" \
--data_dir "./data/meld/MELD_features_raw1.pkl" \
--log_dir ${LOG_PATH}/${EXP_NO} \
--nodal-attention \
--lr 0.0003 \
--l2 0.00001 \
--gamma 0.1 \
--beta 0.0 \
--dropout 0.3 \
--tau 1 \
--modulation \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
