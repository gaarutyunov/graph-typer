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
#SBATCH --job-name=gt-eval
#SBATCH --output="logs/slurm/eval/"%j.out
#SBATCH --error="logs/slurm/eval/"%j.err
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=3-0:0

# Load modules
chmod +x ./scripts/prepare.sh
source ./scripts/prepare.sh

# Train model
chmod +x ./scripts/eval.sh

DATASET_NAME=${1:-pldi2020}
DATASET_ROOT=${2:-~/data}
MODEL_ARCH=${3:-graph_coder_masked_base}
CKPTS_PATH=${4:-${DATASET_NAME}-${MODEL_ARCH}}
TOP_N=${5:-1}
NUM_CLASSES=${6:-1000}
MAX_TOKENS=${7:-8192}
OUTPUT_PREDICTIONS=$8

echo "Evaluating Top-$TOP_N..."
echo "Model arch: $MODEL_ARCH"
echo "Dataset name: $DATASET_NAME"
echo "Dataset root: $DATASET_ROOT"
echo "Max tokens: $MAX_TOKENS"
echo "Using checkpoints from: ckpts/$CKPTS_PATH"
echo "Num classes: $NUM_CLASSES"

source ./scripts/eval.sh "$DATASET_NAME" "$DATASET_ROOT" "$MODEL_ARCH" "$CKPTS_PATH" "$TOP_N" "$NUM_CLASSES" "$MAX_TOKENS" "$OUTPUT_PREDICTIONS"