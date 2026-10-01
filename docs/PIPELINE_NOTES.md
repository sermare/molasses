# Pipeline notes (moved out of the README)

How to run and resume the pipeline on the Savio cluster, the orchestrators, the CPU baselines, and the controls and caveats that were written
when only target 588689 had finished (so statements such as "single target" or "588689 is the easiest" refer to that stage; the current
status and findings are in the README).

## How to run / resume

```bash
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh 588689

# Pass-1 (143 chunks); resume = only missing chunks (see slurm/ for the exact resubmit pattern)
sbatch --export=ALL,TARGET=$TARGET --array=1-143%40 slurm/pass1_embed.sbatch
# ...consolidation auto-runs; then:
sbatch --export=ALL,TARGET=$TARGET --array=1-143%40 slurm/pass2_affcache.sbatch
sbatch --export=ALL,TARGET=$TARGET slurm/score_base.sbatch
sbatch --export=ALL,TARGET=$TARGET --array=1-15    slurm/train_score_seed.sbatch
python pipeline_local/eval_headft_seeds.py --targets $TARGET
```

Scripts are **resume-friendly** (skip existing checkpoints/scores). All boltz jobs write logs to
`slurm/logs/`.

### Orchestrators (CPU, non-GPU node, `--requeue`, idempotent)
Two exist; both run on `savio3_htc` lowprio and survive session ends/preemption:

- **`slurm/finish_pipeline.sh`** (via `slurm/finish.sbatch`) — **the one in use.** Starts Pass-2 on
  the complete chunks *without waiting for all of Pass-1* (Pass-2 only needs per-chunk poses), then
  drives tail Pass-2 → full merge → `build_ft_inputs` → base + 15 FT arms → eval. Use this.
- **`slurm/driver.sbatch`** (`slurm/pipeline_driver.sh`) — the original all-or-nothing driver;
  superseded because it blocked consolidation until 100% of Pass-1 and thrashed under preemption.
  Kept for reference / clean single-target runs from scratch.

```bash
sbatch slurm/finish.sbatch                                 # resumes an in-progress run to completion
# or, a clean run from scratch:
sbatch --export=ALL,TARGET=588689 slurm/driver.sbatch
```

Both end at the evaluation → `results/analysis/<TARGET>/{per_seed,spread,table2}.csv`.

## CPU baselines (no Boltz)

Supervised descriptor baselines that run entirely on CPU, in parallel with the GPU pipeline —
they need only SMILES + labels + the ranking, so they don't wait on folding:
`slurm/cpu_baselines.sbatch` → `pipeline_local/cpu_baselines.py`. Trained on the same top-N
compounds (N=40/100/300) and evaluated on the same common eval set as head-FT.

- **LightGBM on 2048-bit ECFP** (Morgan r2) — CV-tuned (27-combo grid, 3-fold stratified, max AP),
  5 seeds + a 5-model average (paper-style).
- **Nearest-active Tanimoto** — rank the eval set by max ECFP similarity to a training active
  (the simplest chemical-similarity baseline; no training).

```bash
sbatch --export=ALL,TARGET=588689 slurm/cpu_baselines.sbatch   # → results/analysis/cpu_baselines_<target>.csv
```

Results slot into `02_benchmarking` alongside No-FT and (later) head-FT. CheMeleon / DrugCLIP
baselines are out of scope for now (need chemprop + model downloads).

### Controls and honest caveats (read before quoting any number)
- **Provisional / single target.** All current numbers are 588689 only, which is the *easiest* of
  the eight (highest No-FT AP/EF) and per the SI the one target where head-FT *reduced* unseen-scaffold
  recovery. Do not generalize from it; more targets are needed for any claim (held per user).
- **`score_boltz2` is the INITIAL ranking, not No-FT.** The paper distinguishes the initial ranking
  (pose reselection) from the No-FT ranking (stored pre-affinity poses rescored). Our baseline uses
  `score_boltz2` as a proxy; head-FT must be compared to the pipeline's own base scoring, not this.
- **Stratified analysis uses recovery from the full-eval ranking** (paper convention, ties by eval-ID
  order) — NOT stratum-relative EF, which inflates by shrinking the base rate (full 0.80% / unseen
  0.66% / low-sim<0.3 0.49%). Stratifying by "scaffold absent from the 300 training compounds" only
  affects label-using methods, so a label-free method is near-flat by construction — this is an
  analog-bias probe on the baselines, not a fair No-FT-vs-labels test.
- **Bootstrap 95% CIs** are attached to every full-eval metric (`cpu_ci_*.csv`); single-run point
  differences on one target are not distinguishable from noise.
- **The real head-FT bar** (unseen stratum, matched N=300, with CIs): vs pipeline No-FT; vs
  nearest-active at matched N; and vs the low-similarity (max Tc<0.3) stratum — the hardest, where the
  paper's own effect is weakest. The contribution of this replication is CIs + seeds + a non-easiest
  target, not the controls themselves.

