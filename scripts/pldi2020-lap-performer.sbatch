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
#SBATCH --job-name=gt-train
#SBATCH --output="logs/slurm/train/"%j.out
#SBATCH --error="logs/slurm/train/"%j.err
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=7-0:0

# Load modules
cd scripts

chmod +x ./prepare.sh
source ./prepare.sh

# Train model
chmod +x ./pldi2020-lap-performer.sh

DATASET_ROOT=${1:-~/data}
MODEL_ARCH=${2:-graph_coder_base}
MODEL_NAME=${3:-pldi2020-gc-lap-performer-${MODEL_ARCH}}
BATCH_SIZE=${4:-16}

echo "Training..."
echo "Model arch: $MODEL_ARCH"
echo "Dataset root: $DATASET_ROOT"
echo "Saving checkpoints to: ckpts/$MODEL_NAME"
echo "Batch size: $BATCH_SIZE"

source ./pldi2020-lap-performer.sh "$DATASET_ROOT" "$MODEL_ARCH" "$MODEL_NAME" "$BATCH_SIZE"