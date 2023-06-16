#!/usr/bin/env bash

## FMNIST-PS
# CKPT_PATH="${ASSETDIR}/fspace-inference/sweep-fomumr3v/run-20230430-xsi20ky2/files"

## FMNIST-FSGC
# CKPT_PATH="${ASSETDIR}/fspace-inference/run-20230120_135117-5u14sge6/files"

## FMNIST-LMAP
# CKPT_PATH="${ASSETDIR}/fspace-inference/sweep-fomumr3v/run-20230430-9ssmghf3/files"

## C10-PS
CKPT_PATH="${ASSETDIR}/fspace-inference/sweep-hen90pon/run-20230507-dy3hlo7x/files"

## C10-noaug-PS
# CKPT_PATH="${ASSETDIR}/fspace-inference/sweep-yb863d1n/run-20230502-ykv04phz/files"

## C10-FSGC
# CKPT_PATH="${ASSETDIR}/fspace-inference/run-20230124_113748-j7qkmtb7/files"

## C10-LMAP
# CKPT_PATH="${ASSETDIR}/fspace-inference/sweep-hen90pon/run-20230507-31tqatuj/files"

## C10-noaug-LMAP
# CKPT_PATH="${ASSETDIR}/fspace-inference/sweep-yb863d1n/run-20230502-vq47i2sh/files"

## TWOMOONS-PS
# CKPT_PATH=".log/psmap"

## TWOMOONS-FS
# CKPT_PATH=".log/pathology"

seedlist=(9 99 999 9999 99999)

for SEED in ${seedlist[@]}; do

python experiments/evaluate_landscape.py \
    --dataset=cifar10 \
    --model-name=resnet18 \
    --batch-size=128 \
    --ckpt-path=${CKPT_PATH} \
    --step_lim=50. \
    --n-steps=50 \
    --seed=${SEED}

done
