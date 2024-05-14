#!/bin/bash
#SBATCH --job-name=gt-data           # Название задачи
#SBATCH --output="logs/slurm/data/"%j.out
#SBATCH --error="logs/slurm/data/"%j.err
#SBATCH --time=3-0:00               # Максимальное время выполнения
#SBATCH --ntasks=1                   # Количество MPI процессов
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --cpus-per-task=10            # Требуемое кол-во CPU

DATASET_NAME=${1:-pldi2020}
DATASET_ROOT=${2:-~/data}
SPLIT=${3:-train}
MAX_TOKENS=${4:-4096}
NUM_CLASSES=${5:-1000}
PROCESSED_DIR=${6:-"processed-data-$MAX_TOKENS"}

echo "Processing dataset $DATASET_NAME"
echo "Root: $DATASET_ROOT"
echo "Processed dir: $PROCESSED_DIR"
echo "Split: $SPLIT"
echo "Max tokens: $MAX_TOKENS"
echo "Num classes: $NUM_CLASSES"
echo "Num workers: $SLURM_CPUS_PER_TASK"

PYTHONPATH=. python -m graph_coder.data \
--dataset-name "$DATASET_NAME" \
--dataset-root "$DATASET_ROOT" \
--processed-dir "$PROCESSED_DIR" \
--split "$SPLIT" \
--max-tokens "$MAX_TOKENS" \
--num-classes "$NUM_CLASSES" \
--num-data-workers "$SLURM_CPUS_PER_TASK"
