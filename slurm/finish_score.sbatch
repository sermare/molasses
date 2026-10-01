#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=bft_fscore
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/fscore_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/fscore_%j.err
echo "[fscore.sbatch] start/resume $(date) restarts=$SLURM_RESTART_COUNT"
bash /global/scratch/users/sergiomar10/boltzaff/slurm/finish_score.sh 588689
