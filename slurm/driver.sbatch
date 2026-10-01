#!/bin/bash
# Long-running CPU driver (non-GPU node). Self-resuming under preemption via --requeue:
# the driver script is idempotent, so a requeued run continues from disk state.
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
#SBATCH --job-name=bft_driver
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/driver_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/driver_%j.err
TARGET="${TARGET:-588689}"
echo "[driver.sbatch] start/resume $(date) on $(hostname) TARGET=$TARGET restarts=$SLURM_RESTART_COUNT"
bash /global/scratch/users/sergiomar10/boltzaff/slurm/pipeline_driver_v4.sh "$TARGET"
