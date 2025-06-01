import numpy as np
from itertools import chain, combinations
from typing import Set, Callable
from scipy.stats import poisson
Po = poisson.pmf


# does not include S or {}
def powerset(iterable):
    s = list(iterable)
    return [set(c) for c in chain.from_iterable(
        combinations(s, r) for r in range(1, len(s))
    )]

class Dir:
    def __init__(self, dim, sign):
        self.dim = dim
        self.sign = sign
    def __eq__(self, other):
        return self.dim == other.dim and self.sign == other.sign
    def __hash__(self):
        return hash((self.dim, self.sign))

DEC, INC = -1, 1
class Transition:
    def __init__(
        self,
        dirs: Set[Dir],
        func: Callable,
        virtual: bool = False,
        fails: Set[Dir] = {}
    ):
        self.dirs = dirs
        self.func = func
        self.virtual = virtual
        self.fails = fails

    def __str__(self):
        return " ".join(f"{d.dim}|{d.sign}" for d in self.dirs)