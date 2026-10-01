#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --job-name=bft_resume
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/resume_%j.out
bash /global/scratch/users/sergiomar10/boltzaff/slurm/resume_after_504329.sh
