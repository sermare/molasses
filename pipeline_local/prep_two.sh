#!/bin/bash
set -eo pipefail
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
ROOT=/global/scratch/users/sergiomar10/boltzaff
for T in 504329 624273-588549; do
  echo "#### $T $(date)"; source "$ROOT/env.sh" "$T"
  [ -s "$ROOT/data/${T}_msa.csv" ] || python "$ROOT/pipeline_local/gen_msa.py" --seq-file "$ROOT/data/${T}_seq.txt" --out-csv "$ROOT/data/${T}_msa.csv" --tmp "$WORKDIR/msa_tmp/$T" 2>&1 | tail -1
  python "$BOLTZFT/pipeline/select_subsets.py" --target "$T" --results-csv "$RESULTS_CSV" --out-dir "$R/selection"
  python "$BOLTZFT/pipeline/build_boltz_inputs.py" --target "$T" --inference-csv "$R/selection/inference_compounds.csv" --msa-csv "$ROOT/data/${T}_msa.csv" --out-yaml-dir "$R/inputs_full"
  python "$BOLTZFT/pipeline/make_chunks.py" --target "$T" --inputs-dir "$R/inputs_full" --chunks-root "$R/inputs_chunks_full" --outputs-root "$R/outputs_chunks_full" --index-file "$R/chunks.tsv"
  echo "[$T] chunks=$(wc -l < "$R/chunks.tsv")"
done
echo "PREP_TWO DONE $(date)"
