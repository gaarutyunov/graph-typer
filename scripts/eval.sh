#!/usr/bin/env bash

ulimit -c unlimited

DATASET_NAME=$1
DATASET_ROOT=$2
MODEL_ARCH=$3
CKPTS_PATH=$4
TOP_N=$5
NUM_CLASSES=$6
PROCESSED_DIR=$7
OUTPUT_PREDICTIONS=$8

if [ -n "$OUTPUT_PREDICTIONS" ]; then
  ARG="--output-predictions"
else
  ARG="--no-output-predictions"
fi

PYTHONPATH=. python graph_coder/evaluate/evaluate.py \
$ARG \
--output-dir ./scripts/ckpts/"$CKPTS_PATH"/predictions \
--split test \
--metric auc \
--top-n "$TOP_N" \
--dataset-name "$DATASET_NAME" \
--dataset-root "$DATASET_ROOT" \
--processed-dir "$PROCESSED_DIR" \
--user-dir graph_coder \
--num-workers 0 \
--ddp-backend=legacy_ddp \
--task node_classification \
--user-data-dir graph_coder/data \
--criterion cross_entropy_loss \
--counter-path "$DATASET_ROOT"/"$PROCESSED_DIR"/train/counter.pkl.gz \
--sizes-path "$DATASET_ROOT"/"$PROCESSED_DIR"/train/sizes.pkl.gz \
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
--batch-size 1 \
--data-buffer-size 20 \
--tensorboard-logdir ./scripts/tb/"$CKPTS_PATH" \
--checkpoint-path ./scripts/ckpts/"$CKPTS_PATH"/checkpoint_best.pt \
--output-path ./scripts/ckpts/"$CKPTS_PATH"/result_top_"$TOP_N".json \
--metadata-path "$DATASET_ROOT"/tensorised-data/train/metadata.pkl.gz \
--type-lattice-path "$DATASET_ROOT"/graph-dataset/_type_lattice.json.gz \
--alias-metadata-path "$DATASET_ROOT"/typingRules.json
