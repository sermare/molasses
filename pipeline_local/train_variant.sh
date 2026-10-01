#!/bin/bash
# Train + score one training-composition variant (extension experiment). Reuses the Pass-2
# affinity cache; only the train-ID membership differs. Eval set = ft_inputs_variants (rank>600).
# Usage: train_variant.sh TARGET COND SEED   (COND like 'top_N300', 'diverse_N600')
set -eo pipefail
TARGET="$1"; COND="$2"; SEED="$3"
N="${COND##*_N}"                                    # budget parsed from the condition name
: "${WORKDIR:?}"; : "${BOLTZ_CACHE:?}"; : "${BOLTZ2_AFFINITY:?}"; : "${BOLTZFT:?}"
R="$WORKDIR/runs/$TARGET"
CONSOL="$R/consolidated_full"; FTIN="$R/ft_inputs_variants"
CFG="$BOLTZ2_AFFINITY/scripts/train/config_aff/config.yaml"
OUTDIR="$R/headft_variants/${COND}_seed${SEED}"
FTDIR="$R/headft_variants/work_${COND}_seed${SEED}"
CACHE="$R/outputs_affcache"
SCORES="$R/headft_variants_scores/${COND}_seed${SEED}"

if [ ! -f "$OUTDIR/checkpoints/last.ckpt" ]; then
  echo "[var] train $COND (N=$N) seed=$SEED $(date)"; rm -rf "$FTDIR" "$OUTDIR"; mkdir -p "$FTDIR"
  tail -20 "$FTIN/eval_ids.txt" > "$FTDIR.val_ids.txt"
  python "$BOLTZFT/pipeline/prepare_ft_dataset.py" \
    --ml-table "$FTIN/ml_table_train.csv" --target "$TARGET" \
    --source-yaml-dir "$CONSOL/yamls" --source-consolidated-dir "$CONSOL" --out-dir "$FTDIR" \
    --train-ids-file "$FTIN/train_ids_${COND}.txt" --val-ids-file "$FTDIR.val_ids.txt"
  SPLIT=$(ls "$FTDIR"/splits/val_explicit_train*.txt | head -1)
  python "$BOLTZ2_AFFINITY/scripts/train/trainv2.py" "$CFG" \
    seed="$SEED" paths=example paths.pretrained="$BOLTZ_CACHE/boltz2_aff.ckpt" \
    paths.base_dir="$FTDIR" paths.mol_dir="$BOLTZ_CACHE/mols" paths.output_dir="$OUTDIR" \
    paths.split_file="$SPLIT" paths.samples_per_epoch="$N" \
    wandb=null save_top_k=1 disable_checkpoint=false trainer.max_epochs=5 trainer.log_every_n_steps=5 \
    data.num_workers=4 data.random_seed="$SEED" data.pad_to_max_tokens=false data.pad_to_max_atoms=false \
    trainer.accumulate_grad_batches=4 data.batch_size=1 \
    model.training_args.affinity_absolute_weight=0.0 model.training_args.affinity_pairwise_weight=0.0 \
    model.training_args.affinity_binary_weight=1.0 model.training_args.max_lr=0.00002 \
    model.training_args.lr_warmup_no_steps=2 model.training_args.lr_start_decay_after_n_steps=20
fi
CK="$OUTDIR/checkpoints/last.ckpt"; [ -f "$CK" ] || { echo "[var] NO CKPT"; exit 1; }
mkdir -p "$SCORES"
echo "[var] score $COND seed=$SEED $(date)"
for split in "$FTIN"/eval_chunks/chunk_*.txt; do
  name="$(basename "$split" .txt)"; out="$SCORES/$name.csv"; [ -f "$out" ] && continue
  python "$BOLTZFT/pipeline/affinity_cached_infer.py" --pretrained "$CK" --cache-dir "$CACHE" \
    --split-file "$split" --out-csv "$out" --no-kernels
done
echo "[var] done $COND seed=$SEED chunks=$(ls "$SCORES"/chunk_*.csv|wc -l) $(date)"
