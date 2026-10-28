"""Pool with overlapping generations, tournament selection and a hall of fame."""
import random
from dataclasses import dataclass, field

import torch

from .model import build
from .reproduction import fertilize, mutate
from .train import evaluate, make_eval_sets, train


@dataclass
class Individual:
    model: torch.nn.Module
    metrics: dict = field(default_factory=dict)
    age: int = 0

    @property
    def fitness(self):
        return self.metrics["fitness"]


class Population:
    def __init__(self, cfg):
        self.cfg, self.ev, self.tr = cfg, cfg["evolution"], cfg["train"]
        torch.manual_seed(cfg["seed"])
        self.rng = random.Random(cfg["seed"])
        self.gen = torch.Generator().manual_seed(cfg["seed"])
        self.eval_sets = make_eval_sets(cfg)
        self.train_len = cfg["task"]["length"]
        self.steps = 0  # total gradient steps, for equal-compute baselines
        self.hall_of_fame = []

        founders = [self._raise(build(cfg)) for _ in range(self.ev["founders"])]
        self.pool = founders + [self._child(founders) for _ in range(self.ev["pop_size"] - len(founders))]

    def _evaluate(self, model):
        return evaluate(model, self.eval_sets, self.train_len)

    def _raise(self, model):
        self.steps += train(model, self.tr["child_steps"], self.cfg, self.gen)
        return Individual(model, self._evaluate(model))

    def _select(self, pool):
        return max(self.rng.sample(pool, min(self.ev["tournament_k"], len(pool))), key=lambda i: i.fitness)

    def _child(self, pool):
        a, b = self._select(pool), self._select(pool)
        model = mutate(fertilize(a.model, b.model, self.gen), self.ev["mutation_std"], self.gen)
        birth = self._evaluate(model)["fitness"]  # does crossover break networks?
        child = self._raise(model)
        child.metrics["birth_fitness"] = birth
        return child

    def step(self):
        for ind in self.pool:
            self.steps += train(ind.model, self.tr["elder_steps"], self.cfg, self.gen)
            ind.metrics = self._evaluate(ind.model)
            ind.age += 1
        children = [self._child(self.pool) for _ in range(self.ev["children_per_round"])]

        elders = []
        for ind in self.pool:
            if ind.age >= self.ev["lifespan"]:
                ind.model.requires_grad_(False)
                self.hall_of_fame.append(ind)
            else:
                elders.append(ind)
        self.pool = sorted(elders + children, key=lambda i: i.fitness, reverse=True)[: self.ev["pop_size"]]
        return self.summary(children)

    def best(self):
        return max(self.pool + self.hall_of_fame, key=lambda i: i.fitness)

    def summary(self, children):
        fit = [i.fitness for i in self.pool]
        best = max(self.pool, key=lambda i: i.fitness)
        return {
            "steps": self.steps,
            "best_fitness": best.fitness,
            "mean_fitness": sum(fit) / len(fit),
            "mean_birth_fitness": sum(c.metrics["birth_fitness"] for c in children) / len(children),
            "hall_of_fame": len(self.hall_of_fame),
            **{f"best_{k}": v for k, v in best.metrics.items() if k != "fitness"},
        }
