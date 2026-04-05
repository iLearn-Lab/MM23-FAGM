#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

EXP_NO="test"
MODALS="al"

echo "MELD, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/MELD/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u ./code/train.py \
--dataset "MELD" \
--data_dir "./data/meld/MELD_features_raw1.pkl" \
--name ${EXP_NO} \
--speaker_weights '0.5-0.5-1.5' \
--Deep_GCN_nlayers 32 \
--modals ${MODALS} \
--lr 0.001 \
--l2 0.0005 \
--focal 1 \
--log_dir ${LOG_PATH}/${EXP_NO} \
--gamma 0.5 \
--beta 0.5 \
--modulation \
--multi_modal \
--dropout 0.4 \
--use_residue \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
