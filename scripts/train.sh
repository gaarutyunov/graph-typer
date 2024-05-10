#!/usr/bin/env bash

ulimit -c unlimited

DATASET_NAME=$1
DATASET_ROOT=$2
MODEL_ARCH=$3
CKPTS_PATH=$4
BATCH_SIZE=$5
NUM_CLASSES=$6

fairseq-train \
--dataset-root "$DATASET_ROOT" \
--user-dir ../graph_coder \
--num-workers 0 \
--ddp-backend=legacy_ddp \
--dataset-name $DATASET_NAME \
--task node_classification \
--user-data-dir ../graph_coder/data \
--criterion cross_entropy_loss \
--arch "$MODEL_ARCH" \
--performer \
--performer-feature-redraw-interval 100 \
--stochastic-depth \
--prenorm \
--num-classes "$NUM_CLASSES" \
--attention-dropout 0.0 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size "$BATCH_SIZE" \
--data-buffer-size 20 \
--save-dir ./ckpts/"$CKPTS_PATH" \
--tensorboard-logdir ./tb/"$CKPTS_PATH" \
--weights-path "$DATASET_ROOT"/processed-data/train/weights.pkl.gz \
--no-epoch-checkpoints \
--validate-interval-updates 1000 \
--nval 100 \
--batch-size-valid 1 \
--patience 5
