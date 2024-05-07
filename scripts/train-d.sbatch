#! /bin/bash
#
# Copyright 2023 German Arutyunov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#SBATCH --mail-user=germanarutyunov@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=gt-train-d
#SBATCH --output="logs/slurm/train/"%j.out
#SBATCH --error="logs/slurm/train/"%j.err
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --gpus-per-node=2
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=1-0:0

# Load modules
cd scripts

chmod +x ./prepare.sh
source ./prepare.sh

# Train model
chmod +x ./train.sh

DATASET_NAME=${1:-pldi2020}
DATASET_ROOT=${2:-~/data}
MODEL_ARCH=${3:-graph_coder_masked_base}
MODEL_NAME=${4:-${DATASET_NAME}-${MODEL_ARCH}}
BATCH_SIZE=${5:-16}
PORT=${6:-12345}

echo "Training..."
echo "Model arch: $MODEL_ARCH"
echo "Dataset name: $DATASET_NAME"
echo "Dataset root: $DATASET_ROOT"
echo "Saving checkpoints to: ckpts/$MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Distributed port: $PORT"

source ./train-d.sh "$DATASET_NAME" "$DATASET_ROOT" "$MODEL_ARCH" "$MODEL_NAME" "$BATCH_SIZE" "$PORT"