#!/usr/bin/env bash

ulimit -c unlimited

DATASET_NAME=$1
DATASET_ROOT=$2
MODEL_ARCH=$3
CKPTS_PATH=$4
BATCH_SIZE=$5
NUM_CLASSES=$6
PROCESSED_DIR=$7

NUM_DATA_WORKERS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS))

fairseq-train \
--dataset-name "$DATASET_NAME" \
--dataset-root "$DATASET_ROOT" \
--processed-dir "$PROCESSED_DIR" \
--user-dir ../graph_coder \
--max-tokens 4096 \
--num-data-workers $NUM_DATA_WORKERS \
--num-workers 0 \
--ddp-backend=legacy_ddp \
--task node_classification \
--user-data-dir ../graph_coder/data \
--criterion cross_entropy_loss \
--counter-path "$DATASET_ROOT"/"$PROCESSED_DIR"/train/counter.pkl.gz \
--index-path "$DATASET_ROOT"/"$PROCESSED_DIR"/train/indexes.pkl.gz \
--arch "$MODEL_ARCH" \
--performer \
--performer-feature-redraw-interval 100 \
--stochastic-depth \
--prenorm \
--num-classes "$NUM_CLASSES" \
--attention-dropout 0.0 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 3000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size "$BATCH_SIZE" \
--data-buffer-size 20 \
--save-dir ./ckpts/"$CKPTS_PATH" \
--tensorboard-logdir ./tb/"$CKPTS_PATH" \
--no-epoch-checkpoints \
--validate-interval-updates 1000 \
--nval 100 \
--save-interval-updates 1000 \
--keep-interval-updates 1 \
--patience 5 \
--find-unused-parameters
