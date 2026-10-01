#!/usr/bin/env python3
"""Generate a Boltz `key,sequence` MSA CSV for a single-chain protein via the
colabfold MMseqs2 server. Mirrors boltz/main.py's monomer path so the output is
byte-compatible with what `boltz predict --use_msa_server` would have written.
"""
import argparse
from pathlib import Path

from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data import const


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-file", type=Path, required=True, help="plain text: one protein sequence")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, required=True)
    args = ap.parse_args()

    seq = "".join(args.seq_file.read_text().split())
    args.tmp.mkdir(parents=True, exist_ok=True)

    unpaired = run_mmseqs2(
        [seq],
        str(args.tmp / "unpaired"),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy",
    )
    lines = unpaired[0].strip().splitlines()
    seqs = lines[1::2][: const.max_msa_seqs]  # sequence lines only; query kept as first
    keys = [-1] * len(seqs)
    csv = ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, seqs)]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.write_text("\n".join(csv) + "\n")
    print(f"query_len={len(seq)} msa_rows={len(seqs)} -> {args.out_csv}")
    print(f"first data row (query): {csv[1][:60]}")


if __name__ == "__main__":
    main()
