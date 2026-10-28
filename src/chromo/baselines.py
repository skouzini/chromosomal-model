"""Baselines: A standard training, B haploid evolution, C evolution without lifetime training."""
import copy

import torch

from .model import build
from .population import Population
from .train import evaluate, make_eval_sets, train


def run_evolution(cfg, log=print):
    pop = Population(cfg)
    history = []
    for r in range(cfg["evolution"]["rounds"]):
        row = {"round": r, **pop.step()}
        history.append(row)
        log(row)
    return history, pop


def run_standard(cfg, steps, log_every=2000, log=print):
    """Baseline A: one model, gradient descent only, for `steps` total (equal compute)."""
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    model, eval_sets = build(cfg), make_eval_sets(cfg)
    history, done = [], 0
    while done < steps:
        done += train(model, min(log_every, steps - done), cfg, gen)
        row = {"steps": done, **evaluate(model, eval_sets, cfg["task"]["length"])}
        history.append(row)
        log(row)
    return history


def haploid(cfg):
    """Baseline B config: single submodels, twice the slots, so the parameter count matches."""
    cfg = copy.deepcopy(cfg)
    cfg["model"]["n_slots"] *= cfg["model"]["ploidy"]
    cfg["model"]["ploidy"] = 1
    return cfg


def no_training(cfg):
    """Baseline C config: same evolution loop, zero lifetime training."""
    cfg = copy.deepcopy(cfg)
    cfg["train"]["child_steps"] = cfg["train"]["elder_steps"] = 0
    return cfg
