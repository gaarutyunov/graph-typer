#!/bin/bash
#SBATCH --job-name=gt-data           # Название задачи
#SBATCH --output="logs/slurm/data/"%j.out
#SBATCH --error="logs/slurm/data/"%j.err
#SBATCH --time=7-01:00               # Максимальное время выполнения
#SBATCH --ntasks=1                   # Количество MPI процессов
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --cpus-per-task=4            # Требуемое кол-во CPU

module load singularity/3.9.0
singularity run --bind $1:/usr/data --writable scripts/typilus-data_latest.sif