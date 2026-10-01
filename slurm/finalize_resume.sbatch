#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --job-name=bft_fin
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/fin_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/fin_%j.err
bash /global/scratch/users/sergiomar10/boltzaff/slurm/finalize_resume.sh
