"""Submodel slots (pairs when ploidy=2) feeding a higher-order model."""
import torch
from torch import nn

from .genome import make_chromosome
from .tasks.sorting import N_TOKENS


class ChromoModel(nn.Module):
    def __init__(self, n_slots, ploidy, sub_hidden, sub_out, head_hidden, max_len):
        super().__init__()
        self.ploidy, self.max_len = ploidy, max_len
        n_in = max_len * N_TOKENS
        self.slots = nn.ModuleList(
            nn.ModuleList(make_chromosome(n_in, sub_hidden, sub_out) for _ in range(ploidy))
            for _ in range(n_slots)
        )
        # each pair also contributes one disagreement scalar
        head_in = n_slots * ploidy * sub_out + (n_slots if ploidy == 2 else 0)
        self.head = make_chromosome(head_in, head_hidden, max_len * N_TOKENS)

    def forward(self, x, return_disagreement=False):
        feats, disagree = [], []
        for slot in self.slots:
            outs = [c(x) for c in slot]
            feats += outs
            if self.ploidy == 2:
                disagree.append((outs[0] - outs[1]).norm(dim=1, keepdim=True))
        logits = self.head(torch.cat(feats + disagree, dim=1)).view(-1, self.max_len, N_TOKENS)
        if return_disagreement:
            d = torch.cat(disagree, 1).sum(1) if disagree else torch.zeros(len(x))
            return logits, d
        return logits


def build(cfg):
    m, t = cfg["model"], cfg["task"]
    return ChromoModel(m["n_slots"], m["ploidy"], m["sub_hidden"], m["sub_out"], m["head_hidden"], t["max_len"])
