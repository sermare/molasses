#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --job-name=bft_posedens
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/posedens_%j.out
bash /global/scratch/users/sergiomar10/boltzaff/slurm/pose_density_all.sh
