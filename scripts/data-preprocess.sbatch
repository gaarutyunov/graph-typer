#!/bin/bash
#SBATCH --job-name=gt-data           # Название задачи
#SBATCH --output="logs/slurm/data/"%j.out
#SBATCH --error="logs/slurm/data/"%j.err
#SBATCH --time=3-0:00               # Максимальное время выполнения
#SBATCH --ntasks=1                   # Количество MPI процессов
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --cpus-per-task=10            # Требуемое кол-во CPU

DATASET_ROOT=${1:-~/data}

module load singularity/3.9.0
singularity exec --bind "$DATASET_ROOT":/usr/data --writable scripts/typilus-env_latest.sif /usr/src/datasetbuilder/scripts/process_data.sh