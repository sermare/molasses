#!/bin/bash
# Release held jobs once a target's finish log reports "Pass-2 submitted".  Usage: release_after.sh <finish.log> <jobid> [<jobid> ...]
LOG="$1"; shift; OUT=/global/scratch/users/sergiomar10/boltzaff/slurm/release_after.log
until grep -q "Pass-2 submitted" "$LOG" 2>/dev/null; do
  grep -qE "FAILED" "$LOG" 2>/dev/null && { echo "[$(date +%T)] finish log reports a failure; NOT releasing" >> $OUT; exit 1; }
  sleep 30; done
for j in "$@"; do scontrol release "$j" >> $OUT 2>&1; scontrol update JobId="$j" Nice=2000 ArrayTaskThrottle=40 >> $OUT 2>&1; echo "[$(date +%T)] released $j" >> $OUT; done
