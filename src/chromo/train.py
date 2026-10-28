"""Lifetime training and exact evaluation."""
import numpy as np
import torch
import torch.nn.functional as F

from .tasks.sorting import make_batch, score


def train(model, steps, cfg, gen=None):
    """Short gradient-descent budget. A fresh Adam each call: optimizer state is not inherited."""
    if steps <= 0:
        return 0
    t, tr = cfg["task"], cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    for _ in range(steps):
        x, y = make_batch(tr["batch_size"], t["length"], t["max_len"], gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return steps


def make_eval_sets(cfg, seed=12345):
    """Fixed eval sets: the training length plus held-out longer lists up to max_len."""
    t = cfg["task"]
    gen = torch.Generator().manual_seed(seed)
    return {n: make_batch(t["eval_size"], n, t["max_len"], gen) for n in range(t["length"], t["max_len"] + 1)}


@torch.no_grad()
def evaluate(model, eval_sets, train_len):
    out = {}
    for n, (x, y) in eval_sets.items():
        logits, d = model(x, return_disagreement=True)
        s = score(logits, y, n)
        out[f"pos_acc_{n}"], out[f"exact_{n}"] = s["pos_acc"], s["exact"]
        if n == train_len and d.std() > 0:
            errors = (logits.argmax(-1)[:, :n] != y[:, :n]).sum(1).float()
            # NaN when there are no errors to correlate with
            out["disagree_err_corr"] = float(np.corrcoef(d.numpy(), errors.numpy())[0, 1]) if errors.std() > 0 else float("nan")
    out["fitness"] = out[f"pos_acc_{train_len}"]
    return out
