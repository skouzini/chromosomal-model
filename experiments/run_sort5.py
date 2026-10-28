"""Run diploid evolution plus baselines A/B/C on sort5; write CSVs and a plot to results/.

    python experiments/run_sort5.py [--config configs/sort5.yaml] [--rounds N]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from chromo.baselines import haploid, no_training, run_evolution, run_standard


def write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sort5.yaml")
    p.add_argument("--rounds", type=int)
    p.add_argument("--out", default="results/sort5")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.rounds:
        cfg["evolution"]["rounds"] = args.rounds
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for name, c in [("diploid", cfg), ("B_haploid", haploid(cfg)), ("C_no_training", no_training(cfg))]:
        print(f"== {name}")
        runs[name], pop = run_evolution(c, log=lambda r: print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))
        write_csv(runs[name], out / f"{name}.csv")
        print("best ever:", pop.best().metrics)

    print("== A_standard")
    runs["A_standard"] = run_standard(cfg, runs["diploid"][-1]["steps"], log=print)
    write_csv(runs["A_standard"], out / "A_standard.csv")

    final = {n: h[-1] for n, h in runs.items()}
    (out / "final.json").write_text(json.dumps(final, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in runs.items():
        key = "fitness" if name == "A_standard" else "best_fitness"
        ax.plot([r["steps"] for r in h], [r[key] for r in h], label=name)
    ax.set(xlabel="gradient steps (compute)", ylabel="per-position accuracy (len 5)", title="sort5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fitness.png", dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
