#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

EXP_NO="test"
MODALS="al"

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
--speaker_weights '3-0-1' \
--Deep_GCN_nlayers 16 \
--lr 0.0003 \
--l2 0.0001 \
--focal 0.5 \
--class_weight \
--log_dir ${LOG_PATH}/${EXP_NO} \
--gamma 0.0001 \
--beta 0.08 \
--modulation \
--multi_modal \
--dropout 0.4 \
--tau 1 \
--use_residue \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
