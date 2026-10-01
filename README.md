# BoltzFT reproduction — head-FT virtual screening

Reproducing **Furui & Ohue, _Adapting Boltz-2 with limited experimental activity data
improves early enrichment in virtual screening_** (arXiv:2609.24302), the **head
fine-tuning** result (Table 2 / Fig 2), on the Savio cluster.

This is a **reproduction of the authors' released pipeline** [`ohuelab/BoltzFT`](https://github.com/ohuelab/BoltzFT)
(which drives a patched fork of [`molecularinformatics/Boltz2_affinity`](https://github.com/molecularinformatics/Boltz2_affinity)),
adapted to this cluster's SLURM/conda — **not** a from-scratch reimplementation.

## What is in this repository

Code, SLURM scripts, notebooks (executed, with outputs) and local patches. **Not included:** the data (`data/`), run outputs (`results/`), model weights/cache (`boltz_cache/`), logs, and the two upstream clones. To rebuild the environment, clone
[`ohuelab/BoltzFT`](https://github.com/ohuelab/BoltzFT) at `c7d5616` and [`molecularinformatics/Boltz2_affinity`](https://github.com/molecularinformatics/Boltz2_affinity) at `bc06a0b` into this directory, apply `patches/*.patch` (our local changes on top of the authors' own
`BoltzFT/patches/boltz2_affinity.patch`), and see `env.sh` and the pipeline section below. Cluster paths in the scripts (`/global/scratch/users/sergiomar10/boltzaff`, SLURM account, `savio_lowprio`) are specific to the Savio system where this was run.

## Headline findings (status 2026-09-30: 2 of 8 targets finished, 588689 and 504329)

Everything below is computed in the notebooks (`notebooks/00`-`07`); each claim says which targets it rests on. Only **two targets** are finished, so no cross-target statistic is meaningful yet and every finding should be read as "holds for these two" until the other six are in.

1. **Head-FT replicates in direction on both targets, with a smaller effect than the paper's.** Average precision (AP), 5 seeds, N=300 labels (`01_replication`):
   - 588689: No-FT 0.097 (paper 0.096) -> head-FT 0.207 +/- 0.015, x2.12 (paper x2.46).
   - 504329: No-FT 0.051 (paper 0.047) -> head-FT 0.174 +/- 0.021, x3.42 (paper x4.76).
   - The dataset's shipped No-FT scores reproduce the paper's No-FT values on all 8 targets (same eval-set size and active count; AP within 0.005). That checks the evaluation, not our own Boltz-2 runs (`00_overview`, `01_replication`).
2. **At the top 1% of the library, head-FT finds more actives and flags fewer wrong compounds** (588689, 497 compounds flagged): 125 actives against 67 for No-FT, 372 false positives against 430 (`03_ft_vs_noft_reranking`).
3. **The gain is concentrated in actives that resemble the training actives.** Binning held-out actives by their highest ECFP4 Tanimoto to a training active: those below 0.3 get no net lift (about as many fall 4x as rise 4x); those at 0.3-0.5 and above 0.5 are promoted strongly (588689 median rank gain x2.2 and x5.0; 504329 x8.2 and x11.4). Inactives show almost no such trend (Spearman 0.07-0.12 against 0.31-0.53 for actives). So head-FT behaves like generalisation to analogs of the 300 labels, not like a model that has learned binding in general. Correlational, two targets, and 504329 has only 28 training actives.
4. **Head-FT does not tell near-identical molecules apart any better than No-FT** (`04_structural_cliffs`; pairs of ECFP4 Tanimoto >= 0.6 where one is active and one inactive; 737 pairs, 2 targets). The active is scored above its inactive twin in 67% of pairs for No-FT and 65% for head-FT (difference -0.02, 95% CI -0.05 to +0.04 over chemical series; chance is 50%). Most twins get almost the same score (at Tanimoto >= 0.7, 230 of 262 pairs are near-ties under the dataset's Boltz-2 score): a fine-tuned model moves a whole chemical series up together, which also raises the number of inactive siblings among the top-1% false positives.
5. **The ranking is largely protein-independent on the one target tested** (588689 decoy rescore, `05_pose_density_and_decoy`; the same 1,600 ligands, 396 active, scored against three proteins): real protein AUROC 0.912, an unrelated protein 0.904 (difference +0.008, 95% interval includes zero), a shuffled-sequence protein with no MSA 0.805. The affinity head's ranking here comes mostly from the ligand; the real protein adds little over an unrelated one. Not tested: whether that ligand signal is memorised chemotypes or general ligand properties. The subset is 25% active, so AUROC/AP are not comparable with the library-wide numbers.
6. **Training-set selection (588689 only, 70 arms, 5 seeds each, `06_training_strategies`):** more data stops helping beyond N=300 (not detected with 5 seeds); a balanced training set (half actives) gives x1.27 over the standard top-N set (p = 0.028) and hard-negative training gives x0.70 (p = 0.012). 9 comparisons were made (Bonferroni threshold 0.0056): none passes, so these are suggestive. Untested on any other target; 504329's candidate pool has only 85 actives (588689: 184), so the same design would not give the same balance.
7. **The Boltz-2 structures are folded and mostly confident, not disordered** (`07_structure_confidence`): mean per-residue pLDDT 82-97 across the 8 targets; only 504329 and 2097 have about 10% of residues below pLDDT 50, and 87% of all such residues are at the chain termini. This is Boltz-2's confidence in its own prediction, not an experimental measure.

Caveats that apply throughout: two finished targets; the paper trained each condition once (we use 5 seeds); intervals for pairs resample chemical series, not pairs (one series made 62% of 588689's cliff pairs); the binary label is the only ground truth (no potency labels exist).

## Scale — why this takes a while

Virtual screening means scoring the *whole* library, and Boltz-2 must build a 3D
protein–ligand complex for every candidate before it can score it. Each compound is
co-folded exactly once (one pose), but there are a lot of them:

```
1 target   = ~50,000 compounds = ~50,000 co-folds = ~50,000 predicted structures
             (588689: 49,985)
per fold    ~26 s on one GPU
per target  49,985 × 26 s      ≈ 361 GPU-hours
            ÷ ~28 GPUs         ≈ 13 h wall-clock (overnight)
8 targets                      ≈ 2,900 GPU-hours total, ~400,000 structures
```

Nothing downstream adds to that structure count: Pass-2 (affinity cache) *reuses* the
Pass-1 poses, and the 5 fine-tuning seeds reuse the cached embeddings — neither re-folds.
So the cost is a one-time preprocessing pass whose size is set by the library, not by the
number of experiments. Fine-tuning itself is trivially fast. The lever for less compute is
**fewer targets**, not fewer folds per target — which is why we validate 588689 first.

**Scope (agreed with the user):**
- Validate on **one target (588689) end-to-end**, then scale to the other 7.
- **Head-FT main result only** — no two-stage rescoring cascade, no LightGBM/DrugCLIP baselines.
- **Multi-seed extension:** run head-FT with **≥5 seeds** per condition, report within-condition
  spread, and re-derive Table 2 from **seed-averaged per-target** values (the paper used seed 0 only).
  If the 8-target scale-out is compute-limited, do **N=300 at minimum**.

---

## Key findings so far (target 588689, full eval n=49,685)

- **Replication holds.** No-FT matches the paper almost exactly (AP 0.097 vs 0.096, BEDROC 0.456
  vs 0.455); head-FT at N=300 gives **AP ×2.12 / EF@1% ×1.85** over No-FT — on top of the paper's
  cross-target 2.14× / 1.77×. Clean across 5 seeds. → `notebooks/archive_588689/08_replication_final.ipynb`.
- **Training-set selection (`notebooks/archive_588689/10_ft_strategies.ipynb`, `12_balanced_training.ipynb`; 70 arms, 5 seeds each).**
  The budget saturates at N=300 (AP 0.155; N=600 and N=1000 are no better). At N=300 a **balanced** set (top-scored
  actives plus random inactives) gives AP ×1.27 over the paper's naive top-N (paired p=0.028, uncorrected for the nine
  strategy comparisons, so promising rather than established; at N=600 it shrinks to ×1.08, p=0.053). **Hard negatives
  hurt**, the firmest result: AP ×0.70 against top-N (p=0.012), and balanced beats hard-negative 0.195 to 0.108 in 5 of 5
  seeds (p<0.0001), because pushing high-scoring decoys down also demotes true actives. Whether top-scored actives beat
  scaffold-diverse ones is borderline (0.195 vs 0.170; paired t-test p=0.044, Wilcoxon p=0.062).
- **Uncertainty is diagnostic, not a screening lever.** Across 20 tests, no uncertainty measure
  (two-head disagreement, cross-seed variance, structural confidence) improves early enrichment or
  flags the model's errors; its value is reproducibility/triage (top-1% shortlist is only ~0.69
  Jaccard across seeds) — and 5-seed **ensembling** lifts AP ~4%. →
  `notebooks/archive_588689/07`, `09`, `notebooks/archive_588689/11_uncertainty_deep.ipynb`.
- **Near-identical molecules (`14_structural_cliffs`, `15_correct_large_deltas`).** 406 pairs where the label flips between
  near-identical molecules, plus 275 both-active pairs; one series of 52 compounds supplies 62% of the flips, so results are
  reported with and without it. The models mostly score near-identical siblings alike; their ability to rank the active
  above its inactive partner (about 0.65 to 0.70) is not distinguishable from picking the larger or more lipophilic
  molecule; losing binding goes with a smaller, lighter, less lipophilic partner; and neither confidence nor pose explains
  it in general. Correct large calls are few (91 of 406 pairs, 27 actives, 30 inactive partners, 86% in one series) and are
  not shared across methods.
- **Head-FT vs No-FT, three ways (`notebooks/archive_588689/16_ft_vs_noft.ipynb`).**
  *Whole library* (49,685 held-out compounds, 396 active): at equal cutoffs head-FT is better on every measure and worse on
  none (top 1%: 125 vs 67 actives found, 372 vs 430 false positives, 271 vs 329 missed). At the default p>0.5 threshold it
  looks conservative (flags 200 compounds against 2,062; finds 75 vs 190 actives; 125 vs 1,872 false positives), which is a
  threshold effect from its lower probabilities, not a worse ranking.
  *Similar molecules* (406 label-flip pairs, 275 both-active pairs, 94 series): head-FT is not shown to be better or worse.
  The active is ranked above its near-identical inactive partner 66.0% vs 67.2% of the time (difference -1.2 points, 95% CI
  -4 to +11), and both models detect only about 13-25% of the flips at 90% specificity.
  *Confidence across the 5 seeds*: the seeds agree on the order for 84% of flip pairs, but agreement is not correctness (27%
  are unanimously wrong). At matched score level, similar molecules are neither unusually confident nor unusually uncertain,
  and none of seed agreement, two-head agreement or pose confidence flags which calls to trust.
- **What fine-tuning changes in the ranking (`notebooks/archive_588689/18_reranked_molecules.ipynb`).** At the top 1% (497 compounds) head-FT holds
  125 actives against 67 and 372 false positives against 430, but that net (+58 actives, -58 false positives) hides 704 compounds
  changing status (76 actives promoted, 18 lost; 334 false positives removed, 276 created). The promoted actives lie much closer to
  the 90 training actives than held-out actives do in general (median highest Tanimoto 0.47 vs 0.28; 28% have an analog at 0.6 or
  above, against 11%), and the new false positives include inactive members of the same families, so part of the gain and of the cost
  is fine-tuning recognising chemical families it was shown (58% of the promoted actives have no training active within 0.5).
  `notebooks/archive_588689/17_per_target_ap.ipynb` shows average precision per target on a log scale (the paper's values for all eight, ours as
  they complete).

---

## Directory layout

```
boltzaff/
├── README.md                 ← this file
├── env.sh                    ← `source env.sh [TARGET]` sets all paths/env vars
├── BoltzFT/                  ← cloned pipeline + eval + data/splits + patch
├── Boltz2_affinity/          ← fork @ bc06a0b + BoltzFT patch applied (editable install)
├── boltz_cache/              ← symlinks to weights + CCD mols (+ downloaded mols.tar)
├── data/                     ← raw CSVs, cleaned *_results.csv, *_seq.txt, *_msa.csv, 3EVG.cif
├── pipeline_local/           ← our added scripts (MSA gen, seed wrappers, multi-seed eval, prep)
├── slurm/                    ← sbatch templates + logs + monitors
├── notebooks/                ← 00_data_cleanup / 00_data_stats / 00_data_prep (executed)
└── results/runs/<TARGET>/    ← per-target artifacts (inputs, chunks, poses, caches, scores)
```

**Environment:** conda env `boltzft` (clone of `boltzba` + `pip install -e Boltz2_affinity --no-deps`).
The fork supplies the custom `boltz predict` flags the pipeline needs
(`--write_embeddings_cropped`, `--skip_affinity_prediction`, `--write_affinity_embeddings_cropped`,
`--affinity_skip_run_structure`, `--reuse_pre_affinity_dir`) — stock boltz lacks them.

**SLURM:** account `co_nilah`, partition `savio3_gpu`, QOS `savio_lowprio`, `--gres=gpu:1`.
Nodes **n0143/n0144 are excluded** (their GPUs fail every job at CUDA init).

---

## Data & provenance

Compound sets + binary labels + cached Boltz-2 scores come from `mf-pcba_test.zip`
(Boltzina v1.0.1 release). Construct sequences are **not** distributed; we reconstructed them:

| Target folder | Protein | Source | Construct len |
|---|---|---|---|
| 588689 | Dengue-2 NS5 MTase | PDB 3EVG | 275 |
| 540297-493091 | SCP1 phosphatase | PDB 3PGL | 180 |
| 434954-2097 | GSK-3β | PDB 1J1B | 420 |
| 463203-2650 | GSK-3α | PDB 7SXF | 345 |
| 493248-485317 | ALR / GFER oxidase | PDB 3MBG | 139 |
| 504329 | Influenza A NS1 | predicted (NCBI GI 227977143) | 219 |
| 624273-588549 | M.tb FadD28 ligase | predicted (NCBI GI 1781172) | 580 |
| 1053173-743445 | NSD2 PWWP1 domain | PDB 6UE6 | 141 |

**Validation:** all 8 targets reproduce the paper's Table 1 (neval, eval actives, train actives at
N=40/100/300) exactly, and 2 PDB assignments were cross-checked against PubChem BioAssay targets.
**Caveat:** exact constructs (esp. His-tag boundaries / predicted-target truncations) are best
reconstructions — a fidelity note, not a validated match. MSAs are generated per target via the
colabfold MMseqs2 server and shared across that target's compounds.

**Cost reality:** Pass-1 co-folding ≈ **26 s/compound** → **~360 GPU-h per target** for the full
~50k-compound eval set. This is *fixed per target* — independent of seed count and N budget (the
embedding cache is shared). Fine-tuning itself is trivially cheap.

---

## Pipeline (annotated steps)

Run per target. `source env.sh <TARGET>` first (sets `$R`, `$BOLTZFT`, `$WORKDIR`, etc.).

1. **Clean data** — map raw columns to the BoltzFT schema (`target_active_v2`→`Active_v2`,
   `score_boltz2`→`affinity_probability_binary`). → `data/<TARGET>_results.csv`
   *(all 8 done; see `pipeline_local/setup_targets.py`)*
2. **MSA** — `pipeline_local/gen_msa.py` → `data/<TARGET>_msa.csv` (boltz `key,sequence` format).
3. **Select subsets** — `BoltzFT/pipeline/select_subsets.py` → nested top-40/100/300 + eval subset.
4. **Build inputs** — `build_boltz_inputs.py` → one YAML per compound (protein+MSA+ligand+affinity).
5. **Chunk** — `make_chunks.py` → 143 chunks × 350 → `chunks.tsv` (one SLURM array task per chunk).
6. **Pass-1 (GPU, expensive)** — `slurm/pass1_embed.sbatch`: `boltz predict
   --write_embeddings_cropped --skip_affinity_prediction`. Produces the predicted **complex
   (`*_model_0.cif`)**, confidence maps, and cropped trunk embeddings for every compound.
7. **Consolidate** — `merge_consolidate.py` + `build_ft_inputs_full.py` → `consolidated_full/`
   (structures, embeddings, manifest) + `ft_inputs_full/` (ml_table, eval_ids, eval_chunks).
   *(auto-runs when Pass-1 finishes — `slurm/mon_consolidate.sh`)*
8. **Pass-2 (GPU, cheaper)** — `slurm/pass2_affcache.sbatch`: reuses Pass-1 poses
   (`--affinity_skip_run_structure --reuse_pre_affinity_dir`) → shared affinity cache.
9. **Multi-seed head-FT (GPU, fast)** — `slurm/train_score_seed.sbatch` (array 1–15 = N{40,100,300}
   × seeds{0–4}). Each: `pipeline_local/train_headft_seed.sh` (binary focal loss, 5 epochs, lr 2e-5,
   the paper's settings + seed) then `pipeline_local/score_arm.sh` over all eval chunks.
10. **Base scoring** — `slurm/score_base.sbatch`: No-FT arm scored once over the eval set.
11. **Evaluate** — `pipeline_local/eval_headft_seeds.py` → `per_seed.csv`, `spread.csv` (within-
    condition mean±sd), `table2.csv` (seed-averaged geo-mean ratios + sign/Wilcoxon/CIs).

---

## Checklist (status as of 2026-09-29)

### ✅ Phase 0 — Setup & data (DONE)
- [x] Fork env `boltzft` built + verified (flags, `crop_embeddings`, `trainv2`)
- [x] `boltz_cache` (weights + CCD mols); bad nodes excluded in every GPU template (n0143, n0144, n0215)
- [x] Data cleaned; split order = `score_boltz2` confirmed; **all 8 match paper Table 1**
- [x] Construct sequences + MSAs for all 8; inputs + 143 chunks for **all 8 targets**

### ✅ Phase 1 — 588689 end-to-end (DONE)
- [x] Pass-1 co-folding (49,985 poses), consolidation, Pass-2 affinity cache (49,985), base scoring + 15 head-FT arms
- [x] Eval → `results/analysis/588689/table2.csv`: No-FT AP 0.097 / EF@1% 16.9 / BEDROC 0.456 (paper 0.096 / 18.2 / 0.455);
      head-FT N=300 AP 0.207 ± 0.015, i.e. ×2.12 (paper cross-target 2.14×). Validated against the paper.

### ✅ Phase 2 — 588689 deliverables (DONE except the decoy-protein rescore, which is running)
- [x] Interactive 3D pose viewer ("NS5 Docking Explorer") and binding-density render ("NS5 Binding Density")
- [x] CPU baselines (LightGBM-ECFP, nearest-active) with memorization controls
- [x] Training-set selection experiment: 14 conditions × 5 seeds = **70 arms, all scored** (notebooks 10, 12)
- [x] Notebooks 00 to 16 (see the list below): data, QC, benchmarking, pose density on the real protein, replication,
      training report, uncertainty (three notebooks), near-identical molecules, FT vs No-FT
- [x] Per-target AP figure on a log absolute scale with the active-rate line (`17_per_target_ap`; fills in as targets finish)
- [x] Post-FT re-ranked molecule view (`18_reranked_molecules`)
- [~] Decoy-protein rescore (gating experiment for the pose-density mechanism claim): **running**, 1,600 compounds scored against the real,
      a shuffled and an unrelated protein; `19_decoy_protein.ipynb` is built automatically when it finishes (`slurm/decoy_finalize.sbatch`)

### 🔄 Phase 3 — 7-target scale-out (RUNNING)
All seven targets are running. **504329** has finished Pass-1 and is consolidating before Pass-2; the other six are folding at lower
queue priority (`--nice 100`, at most 30 concurrent tasks each) so that 504329's later stages are not starved. Pass-1 progress
(structures folded, counted from per-chunk prediction folders) as of 2026-09-29 19:37, just as the six resumed:

| Target | Pass-1 folded | Chunks complete | Consolidated | Pass-2 |
|---|---|---|---|---|
| 504329 | 49,973 / 49,982 | 143 / 143 | yes | Pass-2 running |
| 485317 | 31,207 (62%) | 70 / 143 | no | not started |
| 743445 | 29,813 (60%) | 78 / 143 | no | not started |
| 493091 | 25,280 (51%) | 66 / 143 | no | not started |
| 2097 | 13,673 (27%) | 4 / 143 | no | not started |
| 2650 | 6,754 (14%) | 4 / 143 | no | not started |
| 588549 | 3,325 (7%) | 0 / 143 | no | not started |

- After Pass-1: consolidate → Pass-2 → base + 15 head-FT arms → eval, per target (`slurm/pipeline_driver_v4.sh`).
- [ ] Per-target `table2.csv` for all 7, then the cross-target Table 2 (geo-mean ratios, sign test, Wilcoxon) and figures.
- [ ] Check that the balanced-selection gain and the hard-negative penalty generalise beyond 588689.

**Orchestration notes.** Each target has its own self-resuming driver (`slurm/driver.sbatch` → `pipeline_driver_v4.sh`,
idempotent, `--requeue`, job names scoped per target), started by `slurm/rollout.sbatch`. Two failures found on 2026-09-29
and fixed: (1) node **n0215** fails every GPU job with "No supported gpu backend found", so it is excluded in all templates
and in the driver; (2) the original driver called any chunk with fewer than 345 outputs incomplete, so the smaller last
chunk (282 compounds) was resubmitted forever and Pass-1 could never finish. `pipeline_driver_v2.sh` compares each chunk
with its own size and counts Pass-1 progress from directory listings, which is also much faster.
(3) **Whole-chunk re-folds.** `pass1_embed.sbatch` re-folded all ~350 compounds of a chunk on every rerun (`--override`), so a chunk
missing 8 compounds cost as much as one missing 350 (1.6 h). It now folds only the missing compounds into a separate
`<chunk>_tail_<job>` results folder (consolidation globs `chunk_*/boltz_results_*`, so it is picked up) and links them into the chunk's
predictions folder for the driver's completeness check; the 8-compound tail of 504329 took 3 min 20 s. (4) `pipeline_driver_v3.sh`
adds `--nice 100` and a per-target array cap of 30 (override with `DRIVER_NICE` and `ARRAY_CONC`); `driver.sbatch` now points to v4.
(5) **Unfolded compounds.** `build_ft_inputs_full.py` (the authors' script) crashed with a KeyError on the 9 compounds of 504329 that never
folded, and the old driver carried on and printed DONE after skipping Pass-2 and scoring. The script is locally patched to skip compounds
without a structure (marked in the file; none of the 9 is in the training top 300, all are inactive; the evaluation set is 49,673
against the paper's 49,682, with the paper's 438 actives), and `pipeline_driver_v4.sh` stops with an error if the inputs or the base
scoring are missing instead of continuing. 504329 was relaunched from its existing consolidation, and Pass-2 (143 tasks) is running.

---

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

## Notebooks
`notebooks/` holds the **self-filling** notebooks (run/edit with the registered `boltzba` kernel; each has a `build_*.py`). They loop over all 8 targets and
use `pipeline_local/bft_common.py`: a target is analysed once its No-FT arm and all 15 head-FT arms (3 budgets x 5 seeds) are fully scored; the others print
`pending: <stage>` and fill in on the next re-run, with no edits. Every cross-target statement prints `n targets = k`.

| Notebook | What it covers |
|---|---|
| `00_overview` | Pipeline status per target, ours vs paper headline table, index |
| `01_replication` | Table 2 (geo-mean ratios, sign test, Wilcoxon), AP per target on a log scale, seed spread, paper vs ours |
| `02_uncertainty` | Cross-seed SD, two-head disagreement, matched-score confidence value, calibration, top-1% seed agreement, confident subsets |
| `03_ft_vs_noft_reranking` | Library-level actives / false positives / misses at matched budgets; what head-FT promotes and loses |
| `04_structural_cliffs` | Near-identical pairs, structural reasons, correct large deltas, FT vs No-FT on similar pairs (series-level bootstrap) |
| `05_pose_density_and_decoy` | Ray-traced PyMOL renders of the pose density on the real protein (overview, cut-away, pocket, back; all 8 targets; PyMOL 3.1 in the `pymolenv` conda env, script `pipeline_local/render_density_pymol.py`, PNGs cached in `results/analysis/<t>/render/`); per-residue density; decoy-protein rescore (588689 only) |
| `06_training_strategies` | Training-set selection; balanced / hard-negative variants (variant arms exist for 588689 only) |

The earlier single-target notebooks (00-19, written while only 588689 had finished), their builders and the data-exploration notebooks are kept unchanged in
`notebooks/archive_588689/` (do not re-run their builders; they write to the parent folder).

Analysis scripts behind them are in `pipeline_local/` (`bft_common.py` (shared loader), `bft_replication.py`, `bft_uncertainty.py`, `bft_rerank.py`, `bft_cliffs.py`, `bft_training.py`, `prep_cliffs.py`, `cliff_pairs.py`, `cliff_scores_eval.py`, `seed_scores_eval.py`,
`ft_vs_noft_similar_pairs.py`, `eval_variants.py`, `pose_density_residue.py`, `decoy_inputs.py`, `decoy_harvest.py`).


molecule size bias from , there fore increasing contacts -> boltz2 affinity head ., rosseta style energy fxn 