#!/bin/bash
# One-shot setup of everything this repository needs but does not contain: the authors' code at pinned commits, the Boltz-2 weights,
# the per-target inputs, and (optionally) the conda environments. Safe to re-run; every step checks what is already there.
#
#   scripts/bootstrap.sh --data-zip /path/to/mf-pcba_test.zip            # or an https URL
#   scripts/bootstrap.sh --skip-weights --skip-env --data-zip ...        # code and data only
#   scripts/bootstrap.sh --dry-run                                       # print the steps, change nothing
#
# Steps: 1 clone ohuelab/BoltzFT at c7d5616 and apply patches/BoltzFT_local_changes.patch
#        2 clone molecularinformatics/Boltz2_affinity at bc06a0b and apply the authors' patch (BoltzFT/patches/boltz2_affinity.patch)
#        3 download the Boltz-2 weights (boltz2_conf.ckpt, boltz2_aff.ckpt, mols.tar from Hugging Face boltz-community/boltz-2) into boltz_cache/,
#          check file sizes against the server and extract mols.tar completely (45,227 entries expected)
#        4 unpack the MF-PCBA zip, then write data/<target>_seq.txt and data/<target>_results.csv (pipeline_local/setup_targets.py) and copy the exact MSAs
#        5 create the conda environments from envs/*.yml (boltzba, then boltzft = clone of boltzba + the fork installed with --no-deps)
# After this, run `python scripts/relocate.py ...` once to adapt paths and Slurm settings to your site (see that script's help).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=0; SKIP_W=0; SKIP_E=0; ZIP=""
BOLTZFT_COMMIT=c7d5616; FORK_COMMIT=bc06a0b; HF=https://huggingface.co/boltz-community/boltz-2/resolve/main
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1;; --skip-weights) SKIP_W=1;; --skip-env) SKIP_E=1;; --data-zip) ZIP="$2"; shift;;
  -h|--help) sed -n 2,17p "${BASH_SOURCE[0]}"; exit 0;; *) echo "unknown option $1" >&2; exit 2;; esac; shift; done
run() { echo "+ $*"; [ "$DRY" = 1 ] || eval "$@"; }
cd "$ROOT"

# 1-2. authors' code, pinned
if [ ! -d BoltzFT/.git ]; then run "git clone https://github.com/ohuelab/BoltzFT.git BoltzFT && git -C BoltzFT checkout -q $BOLTZFT_COMMIT"; fi
if [ -d BoltzFT/.git ] && git -C BoltzFT diff --quiet -- pipeline/build_ft_inputs_full.py 2>/dev/null; then run "git -C BoltzFT apply '$ROOT/patches/BoltzFT_local_changes.patch'"; else echo "BoltzFT patch already applied (or BoltzFT not cloned in dry run)"; fi
if [ ! -d Boltz2_affinity/.git ]; then
  run "git clone https://github.com/molecularinformatics/Boltz2_affinity.git Boltz2_affinity && git -C Boltz2_affinity checkout -q $FORK_COMMIT"
  run "git -C Boltz2_affinity apply '$ROOT/BoltzFT/patches/boltz2_affinity.patch'"
else echo "Boltz2_affinity already present"; fi

# 3. weights
if [ "$SKIP_W" = 0 ]; then
  run "mkdir -p boltz_weights"
  for f in boltz2_conf.ckpt boltz2_aff.ckpt mols.tar; do
    want=$(curl -sIL "$HF/$f" | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2}' | tail -1 || true)
    if [ "$DRY" = 1 ]; then echo "+ download $f (expected size from server: ${want:-unknown})"; continue; fi
    [ -s "boltz_weights/$f" ] && [ "$(stat -c %s "boltz_weights/$f")" = "${want:-x}" ] || curl -L -C - -o "boltz_weights/$f" "$HF/$f"
    got=$(stat -c %s "boltz_weights/$f"); [ -z "${want:-}" ] || [ "$got" = "$want" ] || { echo "size mismatch for $f: got $got, expected $want" >&2; exit 1; }
  done
  run "mkdir -p boltz_cache && ln -sfn '$ROOT/boltz_weights/boltz2_conf.ckpt' boltz_cache/boltz2_conf.ckpt && ln -sfn '$ROOT/boltz_weights/boltz2_aff.ckpt' boltz_cache/boltz2_aff.ckpt"
  if [ "$DRY" = 0 ]; then
    n_tar=$(tar tf boltz_weights/mols.tar | wc -l); n_now=0; [ -d boltz_cache/mols ] && n_now=$(( $(ls boltz_cache/mols | wc -l) + 1 ))
    if [ "$n_now" -lt "$n_tar" ]; then tar xf boltz_weights/mols.tar -C boltz_cache; fi     # an interrupted extraction breaks every fold with "KeyError: 'THR'"
    n_now=$(( $(ls boltz_cache/mols | wc -l) + 1 )); [ "$n_now" = "$n_tar" ] || { echo "mols extraction incomplete: $n_now of $n_tar entries" >&2; exit 1; }
    [ -e boltz_cache/mols/THR.pkl ] || { echo "boltz_cache/mols/THR.pkl missing" >&2; exit 1; }
  else echo "+ extract mols.tar into boltz_cache/ and check the entry count"; fi
fi

# 4. data
if [ -n "$ZIP" ]; then
  run "mkdir -p data"
  if [[ "$ZIP" == http* ]]; then run "curl -L -o data/mf-pcba_test.zip '$ZIP'"; ZIPF=data/mf-pcba_test.zip; else ZIPF="$ZIP"; fi
  run "unzip -oq '$ZIPF' -d data/_unzip && mkdir -p data/mf-pcba_test && find data/_unzip -name '*.csv' -exec cp {} data/mf-pcba_test/ \\; && rm -rf data/_unzip"
  run "python pipeline_local/setup_targets.py"
  for t in 588689 504329 1053173-743445 493248-485317 434954-2097 540297-493091 463203-2650 624273-588549; do
    run "gunzip -c inputs/msa/${t}_msa.csv.gz > data/${t}_msa.csv"
  done
else echo "no --data-zip given: skipping data setup (the MF-PCBA zip comes from the Boltzina v1.0.1 release)"; fi

# 5. environments
if [ "$SKIP_E" = 0 ]; then
  command -v conda >/dev/null || { echo "conda not found: skipping environments (see envs/*.yml)"; exit 0; }
  conda env list | grep -q '^boltzba ' || run "conda env create -n boltzba -f envs/boltzba.yml"
  conda env list | grep -q '^boltzft ' || run "conda create -y -n boltzft --clone boltzba && conda run -n boltzft pip install -e '$ROOT/Boltz2_affinity' --no-deps"
fi
echo "bootstrap done. Next: python scripts/relocate.py --help"
