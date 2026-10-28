import torch

from chromo.baselines import haploid
from chromo.genome import crossover, make_chromosome
from chromo.model import build
from chromo.reproduction import fertilize, meiosis
from chromo.tasks.sorting import make_batch, score
import yaml

CFG = yaml.safe_load(open("configs/sort5.yaml"))


def neuron(c, i):
    return torch.cat([c[0].weight[i], c[0].bias[i : i + 1], c[2].weight[:, i]])


def assert_whole_neurons(child, a, b):
    assert child[0].weight.shape == a[0].weight.shape
    assert child[2].weight.shape == a[2].weight.shape
    sources = set()
    for i in range(a[0].out_features):
        n = neuron(child, i)
        from_a, from_b = torch.equal(n, neuron(a, i)), torch.equal(n, neuron(b, i))
        assert from_a or from_b, f"neuron {i} is not whole from one parent"
        sources.add("a" if from_a else "b")
    return sources


def test_crossover_keeps_neurons_whole():
    gen = torch.Generator().manual_seed(0)
    a, b = make_chromosome(20, 64, 7), make_chromosome(20, 64, 7)
    assert assert_whole_neurons(crossover(a, b, gen), a, b) == {"a", "b"}


def test_gamete_is_valid_network():
    model = build(CFG)
    pair = model.slots[0]
    gamete = meiosis(pair)
    assert_whole_neurons(gamete, pair[0], pair[1])
    x, _ = make_batch(4, 5, 8)
    assert gamete(x).shape == pair[0](x).shape


def test_fertilize_shapes_match_and_run():
    for cfg in (CFG, haploid(CFG)):
        a, b = build(cfg), build(cfg)
        child = fertilize(a, b)
        assert [p.shape for p in child.parameters()] == [p.shape for p in a.parameters()]
        x, y = make_batch(4, 5, 8)
        assert child(x).shape == (4, 8, 11)


def test_fertilize_diploid_takes_one_gamete_from_each_parent():
    a, b = build(CFG), build(CFG)
    child = fertilize(a, b)
    for slot, sa, sb in zip(child.slots, a.slots, b.slots):
        assert_whole_neurons(slot[0], sa[0], sa[1])
        assert_whole_neurons(slot[1], sb[0], sb[1])


def test_score_is_exact():
    _, y = make_batch(3, 5, 8)
    logits = torch.nn.functional.one_hot(y, 11).float()
    assert score(logits, y, 5) == {"pos_acc": 1.0, "exact": 1.0}
    x, y_sorted = make_batch(100, 5, 8)
    assert (y_sorted[:, :5].diff(dim=1) >= 0).all()
