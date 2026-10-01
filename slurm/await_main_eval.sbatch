#!/bin/bash
# Durable driver for the main head-FT eval: waits for base+15 FT scoring, backfills, runs
# eval_headft_seeds.py -> table2.csv. Survives session teardown; self-resumes via --requeue
# (await_main_eval.sh is idempotent + resume-friendly).
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=bft_await
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/await_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/await_%j.err
TARGET="${TARGET:-588689}"
echo "[await.sbatch] start/resume $(date) on $(hostname) TARGET=$TARGET restarts=${SLURM_RESTART_COUNT:-0}"
bash /global/scratch/users/sergiomar10/boltzaff/slurm/await_main_eval.sh "$TARGET"
