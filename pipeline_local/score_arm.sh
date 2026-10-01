#!/bin/bash
# Score ALL eval chunks for one arm using the shared affinity cache.
# Usage: score_arm.sh base   TARGET
#        score_arm.sh headft TARGET N SEED
set -eo pipefail
KIND="$1"; TARGET="$2"; N="$3"; SEED="$4"
: "${WORKDIR:?}"; : "${BOLTZ_CACHE:?}"; : "${BOLTZFT:?}"
R="$WORKDIR/runs/$TARGET"
CACHE="$R/outputs_affcache"
INFER="$BOLTZFT/pipeline/affinity_cached_infer.py"
[ -d "$CACHE" ] || { echo "[score] cache missing: $CACHE" >&2; exit 2; }

if [ "$KIND" = "base" ]; then
  CK="$BOLTZ_CACHE/boltz2_aff.ckpt"
  SCORES_DIR="$R/headft_affcache/base/scores"
else
  CK="$R/headft_full/top${N}_seed${SEED}/checkpoints/last.ckpt"
  SCORES_DIR="$R/headft_affcache/lightning_top${N}_seed${SEED}/scores"
  [ -f "$CK" ] || { echo "[score] ckpt missing: $CK" >&2; exit 1; }
fi
mkdir -p "$SCORES_DIR"
echo "[score] kind=$KIND target=$TARGET N=$N seed=$SEED ck=$CK start=$(date)"
for split in "$R"/ft_inputs_full/eval_chunks/chunk_*.txt; do
  name="$(basename "$split" .txt)"
  out="$SCORES_DIR/$name.csv"
  [ -f "$out" ] && continue   # resume-friendly
  python "$INFER" --pretrained "$CK" --cache-dir "$CACHE" \
    --split-file "$split" --out-csv "$out" --no-kernels
done
echo "[score] done kind=$KIND N=$N seed=$SEED chunks=$(ls "$SCORES_DIR"/chunk_*.csv | wc -l) $(date)"
