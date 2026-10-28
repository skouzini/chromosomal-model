# Chromosomal Models

Full plan: `PLAN.md`. Question: do models built from **chromosomal pairs** of submodels, bred by neuron-level crossover plus brief lifetime training, beat (A) standard training, (B) haploid evolution, (C) evolution without lifetime training, at equal compute?

## Key decisions (v1)
- Gene = one hidden neuron of a chromosome (row of W1, bias, column of W2) — always inherited whole. Chromosome = `Linear -> ReLU -> Linear` (`genome.py`).
- Meiosis splices a pair's two members into one gamete; child pair = gamete(A) + gamete(B) (`reproduction.py`).
- Both pair outputs are concatenated into the higher-order model plus one disagreement scalar (L2 of their difference) per pair (`model.py`).
- Higher-order model is inherited by neuron-level crossover of both parents' heads (open question in PLAN.md).
- Haploid baseline = 2x slots, ploidy 1 → same parameter count.
- Compute = total gradient steps (`Population.steps`); baseline A trains for the diploid run's step count.
- Task: sort lists of 5 digits; inputs/outputs padded to `max_len` 8 with a blank token so lengths 6–8 are a held-out eval.

## Commands
- Tests: `python3 -m pytest -q`
- Experiment: `PYTHONPATH=src python3 experiments/run_sort5.py [--rounds N]` → `results/sort5/` (CSVs, final.json, fitness.png)

## Roadmap status
Phases 0–2 scaffolded (code, tests, baselines, runner). Full-size comparison run not yet done.
