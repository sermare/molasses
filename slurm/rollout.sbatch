#!/bin/bash
# Durable master orchestrator for the 7-target rollout. Survives session teardown; self-resumes
# under preemption via --requeue (rollout.sh is idempotent). It only submits/monitors other jobs.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=bft_rollout
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/rollout_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/rollout_%j.err
echo "[rollout.sbatch] start/resume $(date) on $(hostname) restarts=${SLURM_RESTART_COUNT:-0}"
bash /global/scratch/users/sergiomar10/boltzaff/slurm/rollout.sh
