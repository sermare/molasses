# Source this to set up BoltzFT reproduction paths.
# Usage: source /global/scratch/users/sergiomar10/boltzaff/env.sh [TARGET]
export ROOT=/global/scratch/users/sergiomar10/boltzaff
export BOLTZFT=$ROOT/BoltzFT
export BOLTZ2_AFFINITY=$ROOT/Boltz2_affinity
export WORKDIR=$ROOT/results
export BOLTZ_CACHE=$ROOT/boltz_cache
export PYTHONPATH="$BOLTZ2_AFFINITY/src${PYTHONPATH:+:$PYTHONPATH}"
export TARGET="${1:-588689}"
export RESULTS_CSV=$ROOT/data/${TARGET}_results.csv
export MSA_CSV=$ROOT/data/${TARGET}_msa.csv
export R="$WORKDIR/runs/$TARGET"
