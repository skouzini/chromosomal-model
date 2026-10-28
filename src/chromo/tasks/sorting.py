"""Neural sorting: lists of digits 0-9, padded with a blank token to max_len."""
import torch

N_DIGITS = 10
BLANK = 10  # padding token, both in inputs and targets
N_TOKENS = 11


def make_batch(n, length, max_len, gen=None):
    """Return (x one-hot [n, max_len*N_TOKENS], y [n, max_len] token ids)."""
    digits = torch.randint(0, N_DIGITS, (n, length), generator=gen)
    pad = torch.full((n, max_len - length), BLANK)
    xs = torch.cat([digits, pad], dim=1)
    ys = torch.cat([digits.sort(dim=1).values, pad], dim=1)
    x = torch.nn.functional.one_hot(xs, N_TOKENS).float().flatten(1)
    return x, ys


def score(logits, y, length):
    """Exact scoring on the first `length` positions: per-position accuracy and exact-sort rate."""
    pred = logits.argmax(-1)[:, :length]
    correct = pred == y[:, :length]
    return {
        "pos_acc": correct.float().mean().item(),
        "exact": correct.all(1).float().mean().item(),
    }
