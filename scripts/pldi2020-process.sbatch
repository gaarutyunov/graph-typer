#!/bin/bash
#SBATCH --job-name=gt-data           # Название задачи
#SBATCH --output="logs/slurm/data/"%j.out
#SBATCH --error="logs/slurm/data/"%j.err
#SBATCH --time=7-01:00               # Максимальное время выполнения
#SBATCH --ntasks=1                   # Количество MPI процессов
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --cpus-per-task=1            # Требуемое кол-во CPU

DATASET_NAME=${1:-pldi2020}
DATASET_ROOT=${2:-~/data}
SPLIT=${3:-train}
MAX_NODES=${4:-512}
MAX_EDGES=${5:-2048}

echo "Processing dataset $DATASET_NAME"
echo "Root: $DATASET_ROOT"
echo "Split: $SPLIT"
echo "Max nodes: $MAX_NODES"
echo "Max edges: $MAX_EDGES"

PYTHONPATH=. python -m graph_coder.data \
--dataset-name "$DATASET_NAME" \
--dataset-root "$DATASET_ROOT" \
--split "$SPLIT" \
--max-nodes "$MAX_NODES" \
--max-edges "$MAX_EDGES"
