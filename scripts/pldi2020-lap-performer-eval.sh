#!/usr/bin/env bash

ulimit -c unlimited

PYTHONPATH=. python graph_coder/evaluate/evaluate.py --split test \
                           --metric auc \
                           --dataset-root $1 \
                           --user-dir graph_coder \
                           --num-workers 0 \
                           --ddp-backend=legacy_ddp \
                           --dataset-name pldi2020 \
                           --task node_classification \
                           --user-data-dir graph_coder/data \
                           --criterion cross_entropy_loss \
                           --arch graph_coder_base \
                           --lap-node-id \
                           --lap-node-id-k 16 \
                           --lap-node-id-sign-flip \
                           --lap-node-id-eig-dropout 0.2 \
                           --performer \
                           --performer-feature-redraw-interval 100 \
                           --stochastic-depth \
                           --prenorm \
                           --num-classes 100 \
                           --attention-dropout 0.0 --act-dropout 0.1 --dropout 0.0 \
                           --optimizer adam --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
                           --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
                           --lr 2e-4 --end-learning-rate 1e-9 \
                           --batch-size 1 \
                           --data-buffer-size 20 \
                           --tensorboard-logdir ./scripts/tb/pldi2020-gc-lap-performer \
                           --checkpoint-path ./scripts/ckpts/pldi2020-gc-lap-performer/checkpoint_best.pt \
                           --metadata-path ~/data/tensorised-data/train/metadata.pkl.gz \
                           --type-lattice-path ~/data/_type_lattice.json.gz \
                           --alias-metadata-path ~/data/typingRules.json \
                           --weights-path ~/data/weights.pkl.gz
