#!/bin/bash
#SBATCH --job-name=gt-data           # Название задачи
#SBATCH --output="logs/slurm/data/"%j.out
#SBATCH --error="logs/slurm/data/"%j.err
#SBATCH --time=7-01:00               # Максимальное время выполнения
#SBATCH --ntasks=1                   # Количество MPI процессов
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --cpus-per-task=1            # Требуемое кол-во CPU

DATASET_ROOT=${1:-~/data}
SPLIT=${2:-train}

PYTHONPATH=. python -m graph_coder.data --dataset-root "$DATASET_ROOT" --split "$SPLIT"
