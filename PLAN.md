# Chromosomal Models — Project Plan

Oct 27, 2026 · @Josh Skousen

## Overview

The project tests whether models built from **chromosomal pairs** of submodels, bred by genetic crossover, can evolve to solve a verifiable task better than the same architecture trained normally.

Each model is a set of submodel pairs that feed "intuition" to a higher-order model. To reproduce, a model's pair is spliced into a single **gamete**. Two parents' gametes combine to form the child's new pair.

**Research question:** does diploid, crossover-based inheritance plus brief lifetime training beat (a) standard gradient training and (b) the same evolution with single, unpaired submodels?

This is the "nature" half of a larger idea. The generational model communities concept (elders teaching children) is the "nurture" half and is built on top of this foundation in a later phase.

## Core concepts and design decisions

The gene is the **neuron**: its incoming and outgoing weights always travel together, so crossover never creates a half-neuron.

| Concept | Decision for v1 | Later iterations |
| --- | --- | --- |
| Chromosomal pair | Two small MLPs per slot, same shape, both feeding the higher-order model | 3+ chromosomes (polyploidy) as a mutation |
| Gene | One neuron (incoming + outgoing weights), split across indexes within each layer | Layer-level crossover as a comparison |
| Meiosis | Splice the two members of one pair, neuron by neuron, into one gamete | Multiple crossover points, linkage |
| Fertilization | Gamete from parent A + gamete from parent B = child's pair | Mate selection by similarity |
| Pair output | Both members' outputs go to the higher-order model; their disagreement is an uncertainty feature | Dominance (only one member expresses) |
| Mutation | Small Gaussian noise on weights | n±1 neurons per layer, links added or severed (NEAT-style) |
| Lifetime learning | Each child trains briefly with gradient descent before it is scored | Elders teach children (communities phase) |
| Population | Overlapping generations: parents stay in the pool and keep training and breeding for a set number of rounds | Tune lifespan |
| Retirement | Aged-out models are frozen into a benchmark "hall of fame" | Use frozen elders as teachers |
| Fitness | Exact, automatic score on a verifiable task | Multi-task fitness |

Alignment between parents relies on shared ancestry: all models descend from a small set of founders, so the same neuron index keeps roughly the same meaning across the population.

## First experiment: neural sorting

The network itself is the sorter: it takes a list of 5 digits (0–9) and outputs the sorted list. Every answer is checked exactly and instantly, so there is no judge to fool.

**Model**

- 3 chromosomal pairs, each a small MLP over the full input; no aspects are assigned, so we can watch whether pairs specialize on their own.
- A higher-order MLP takes all 6 submodel outputs plus each pair's disagreement and predicts the digit at each of the 5 positions.

**Evolution loop**

1. Seed \~10 founder models, then breed a population of \~50.
2. Each round: pick parents by fitness (tournament selection), make gametes, combine, mutate.
3. Each child trains for a short, fixed budget (e.g. a few hundred steps) before scoring.
4. Parents stay in the pool and keep training for a set number of rounds, then retire to the hall of fame.

**Fitness and metrics**

| Metric | What it tells us |
| --- | --- |
| Per-position accuracy | Partial credit; main fitness signal |
| Exact-sort rate | Share of lists fully sorted |
| Held-out accuracy on longer lists (6–8) | Whether it learned sorting or memorized |
| Within-pair disagreement vs. error | Whether disagreement predicts mistakes |
| Best and mean fitness per round | Whether the population improves |

**Baselines (same parameter count and total compute)**

- **A. Standard training:** the same architecture trained with gradient descent only.
- **B. Haploid evolution:** the same loop with single submodels instead of pairs.
- **C. Evolution without lifetime training:** isolates how much the brief training does.

**Success criteria:** the paired model matches or beats Baseline A at equal compute, beats Baseline B, and shows disagreement that correlates with errors. Any one of these is a useful finding.

## Repo structure and stack

Python 3.11+ with PyTorch, run locally on CPU; experiments are small enough that a GPU is optional.

- **Stack:** PyTorch, NumPy, pytest, YAML configs, matplotlib for plots, plain CSV/JSON logs (optionally Weights & Biases later).
- **Repo:** `chromosomal-models` on GitHub, with a `CLAUDE.md` that summarizes this plan so Claude Code has the context each session.

```
chromosomal-models/
├── CLAUDE.md              # project context for Claude Code
├── README.md
├── pyproject.toml
├── configs/
│   └── sort5.yaml         # population size, mutation rate, budgets
├── src/chromo/
│   ├── genome.py          # neuron-level genes, pairs, gametes
│   ├── model.py           # submodel pairs + higher-order model
│   ├── reproduction.py    # meiosis, fertilization, mutation
│   ├── population.py      # pool, overlapping generations, hall of fame
│   ├── tasks/sorting.py   # data generation + exact scoring
│   ├── train.py           # lifetime training
│   └── baselines.py       # standard + haploid + no-training runs
├── experiments/           # run scripts
├── results/               # logs and plots (git-ignored)
└── tests/                 # crossover keeps neurons intact, shapes match
```

The most important tests check that a gamete is always a valid network: every neuron comes whole from one parent member, and every layer has the right shape.

## Roadmap

Six phases, each gated on a result before the next starts; Phase 2 is the real test of the idea.

&#91;embedded content: roadmap · 6 phases, 5 gates\]

Phases 0–2 are the first milestone. Phases 4 and 5 only start once Phase 3 shows what pairs are actually doing.

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| Crossover breaks networks even with shared ancestry | Compare neuron-level vs. layer-level crossover; measure child fitness right after birth |
| Evolution is far slower than gradient descent | Lifetime training does most of the learning; compare at equal compute |
| Population converges too early (everyone becomes a clone) | Track genetic diversity; raise mutation or add new founders |
| Sorting is too easy, so nothing is learned about the idea | Scale list length and add tasks (reverse, max, dedupe) |
| Size mutations break gene alignment | Defer to Phase 4; use NEAT-style gene IDs |

**Open questions**

- [ ] Should a pair's two outputs be averaged, concatenated, or gated before the higher-order model?
- [ ] Does the higher-order model inherit too, or is it trained fresh for each child?
- [ ] How long should a model stay in the pool before it is frozen?
- [ ] Which external dataset or task comes after sorting?

**References to read**

- NEAT: NeuroEvolution of Augmenting Topologies (Stanley and Miikkulainen, 2002)
- Population Based Training (DeepMind, 2017)
- Git Re-Basin: merging models modulo permutation symmetries (2022)
- Evolutionary Optimization of Model Merging Recipes (Sakana AI, 2024)
