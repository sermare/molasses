#!/bin/bash
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh 588689
echo "[waitP1] waiting for all Pass-1 jobs (bft_p1*) to clear the queue... $(date)"
while squeue -u sergiomar10 -h -n bft_p1,bft_p1_cal -o "%i" 2>/dev/null | grep -q .; do sleep 120; done
echo "[waitP1] Pass-1 queue empty at $(date). Running consolidation."
echo "=== pass1 output chunks with boltz_results ==="
nres=$(find -L "$R/outputs_chunks_full"/chunk_*/boltz_results_* -maxdepth 0 -type d 2>/dev/null | wc -l)
ncropped=$(find -L "$R/outputs_chunks_full" -name 'embeddings_*.npz' 2>/dev/null | wc -l)
echo "result_dirs=$nres  cropped_embeddings_npz=$ncropped (expect ~49985)"
echo "=== merge_consolidate ==="
python "$BOLTZFT/pipeline/merge_consolidate.py" --outputs-root "$R/outputs_chunks_full" \
  --inputs-dir "$R/inputs_full" --out-dir "$R/consolidated_full"
echo "=== build_ft_inputs_full ==="
python "$BOLTZFT/pipeline/build_ft_inputs_full.py" --target "$TARGET" \
  --results-csv "$RESULTS_CSV" --consolidated-dir "$R/consolidated_full" \
  --out-dir "$R/ft_inputs_full"
echo "[waitP1] DONE consolidation $(date)"
echo "structures=$(find -L "$R/consolidated_full/structures" -name '*.npz' 2>/dev/null | wc -l)"
echo "eval_ids=$(wc -l < "$R/ft_inputs_full/eval_ids.txt" 2>/dev/null) eval_chunks=$(ls "$R/ft_inputs_full/eval_chunks"/*.txt 2>/dev/null | wc -l)"
