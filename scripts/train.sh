#!/usr/bin/env bash

ulimit -c unlimited

DATASET_NAME=$1
DATASET_ROOT=$2
MODEL_ARCH=$3
CKPTS_PATH=$4
BATCH_SIZE=$5
NUM_CLASSES=$6
MAX_TOKENS=$7
PORT=$8
PROCESSED_DIR="processed-data-4096"

NUM_DATA_WORKERS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS))

fairseq-train \
--dataset-name "$DATASET_NAME" \
--dataset-root "$DATASET_ROOT" \
--processed-dir "$PROCESSED_DIR" \
--user-dir ../graph_coder \
--max-tokens "$MAX_TOKENS" \
--num-data-workers $NUM_DATA_WORKERS \
--num-workers 0 \
--ddp-backend=pytorch_ddp \
--distributed-port "$PORT" \
--task node_classification \
--user-data-dir ../graph_coder/data \
--criterion focal_loss \
--counter-path "$DATASET_ROOT"/"$PROCESSED_DIR"/train/counter.pkl.gz \
--sizes-path "$DATASET_ROOT"/"$PROCESSED_DIR"/train/sizes.pkl.gz \
--arch "$MODEL_ARCH" \
--performer \
--performer-feature-redraw-interval 100 \
--stochastic-depth \
--prenorm \
--num-classes "$NUM_CLASSES" \
--attention-dropout 0.0 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --weight-decay 0.001 \
--gamma 0.5 \
--clip-norm 25.0 \
--lr-scheduler polynomial_decay --power 0.5 --warmup-updates 3000 --total-num-update 50000 \
--lr 0.1 --end-learning-rate 1e-7 \
--batch-size "$BATCH_SIZE" \
--data-buffer-size 20 \
--save-dir ./ckpts/"$CKPTS_PATH" \
--tensorboard-logdir ./tb/"$CKPTS_PATH" \
--no-epoch-checkpoints \
--patience 10 \
--find-unused-parameters
