"""Neuron-level genes.

A chromosome is a one-hidden-layer MLP (Linear -> ReLU -> Linear). Its genes are the
hidden neurons: neuron i is row i of the first weight, bias i, and column i of the
second weight, and those always travel together. Output biases are inherited per
output unit.
"""
import torch
from torch import nn


def make_chromosome(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out))


@torch.no_grad()
def crossover(a, b, gen=None):
    """New chromosome whose every hidden neuron comes whole from either `a` or `b`."""
    a_in, a_out = a[0], a[2]
    b_in, b_out = b[0], b[2]
    child = make_chromosome(a_in.in_features, a_in.out_features, a_out.out_features)
    c_in, c_out = child[0], child[2]

    from_b = torch.rand(a_in.out_features, generator=gen) < 0.5
    c_in.weight.copy_(torch.where(from_b[:, None], b_in.weight, a_in.weight))
    c_in.bias.copy_(torch.where(from_b, b_in.bias, a_in.bias))
    c_out.weight.copy_(torch.where(from_b[None, :], b_out.weight, a_out.weight))

    bias_from_b = torch.rand(a_out.out_features, generator=gen) < 0.5
    c_out.bias.copy_(torch.where(bias_from_b, b_out.bias, a_out.bias))
    return child
