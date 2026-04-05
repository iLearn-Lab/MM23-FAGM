#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

EXP_NO="test"
MODALS="al"

echo "MELD, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/MELD/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u train.py \
--dataset "MELD" \
--data_dir "./data/MELD_features/MELD_features_raw1.pkl" \
--name ${EXP_NO} \
--modals ${MODALS} \
--lr 0.0003 \
--l2 0.00003 \
--log_dir ${LOG_PATH}/${EXP_NO} \
--multi_modal \
--modulation \
--dropout 0.4 \
--beta 0.8 \
--gamma 0.05 \
--tau 1 \
--use_residue \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
