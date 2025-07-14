import numpy as np
from itertools import chain, combinations
from typing import Set, Callable, List
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
    def mask(self, dirs):
        arr = np.zeros(len(self.dimensions), dtype='int')
        for d in dirs:
            arr[self.dimensions.index(d.dim)] = d.sign
        return arr
    def __init__(
        self,
        dirs: Set[Dir],
        func: Callable,
        dimensions: List[str],
        virtual: bool = False,
        fails: Set[Dir] = {}
    ):
        self.dirs_list = dirs
        self.dimensions = dimensions
        self.dirs = self.mask(dirs)
        self.func = func
        self.virtual = virtual
        self.fails = self.mask(fails)

    def __str__(self):
        return " ".join(f"{d.dim}|{d.sign}" for d in self.dirs_list)