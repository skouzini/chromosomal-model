"""Meiosis, fertilization and mutation."""
import copy

import torch
from torch import nn

from .genome import crossover


def meiosis(pair, gen=None):
    """Splice the two members of one pair, neuron by neuron, into one gamete."""
    return crossover(pair[0], pair[1], gen)


@torch.no_grad()
def fertilize(parent_a, parent_b, gen=None):
    """Child of two parents. Diploid: one gamete from each parent forms each new pair.
    Haploid: each slot is a neuron-level crossover of the parents' chromosomes.
    The higher-order model is inherited the haploid way (crossover of both parents' heads)."""
    child = copy.deepcopy(parent_a)
    for i, (sa, sb) in enumerate(zip(parent_a.slots, parent_b.slots)):
        if parent_a.ploidy == 2:
            child.slots[i] = nn.ModuleList([meiosis(sa, gen), meiosis(sb, gen)])
        else:
            child.slots[i] = nn.ModuleList([crossover(sa[0], sb[0], gen)])
    child.head = crossover(parent_a.head, parent_b.head, gen)
    return child


@torch.no_grad()
def mutate(model, std, gen=None):
    for p in model.parameters():
        p.add_(torch.randn(p.shape, generator=gen) * std)
    return model
