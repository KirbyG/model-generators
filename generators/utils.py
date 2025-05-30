import numpy as np
from itertools import chain, combinations
from typing import Set, Callable
from scipy.stats import poisson
Po = poisson.pmf
def to_shaped(states, shape):
    num_states = np.prod(np.array(shape)+1)
    rank = len(shape)

    P = np.reshape(states[:num_states], np.array(shape)+1)

    extended_shape = np.append(np.array(shape)+1, [rank])
    M = np.reshape(states[num_states:], extended_shape)

    return P, M

def to_linear(states, shape):
    P, M = states
    return np.concatenate((P.flatten(), M.flatten()))

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

def make_mask(dimensions):
    def mask(dirs):
        arr = np.zeros(len(dimensions), dtype='int')
        for d in dirs:
            arr[dimensions.index(d.dim)] = d.sign
        return arr
    return mask

# this code should be simplified using new shaping functions
def extract_mean(vec, axis, shape):
    rank = len(shape)
    P, M = to_shaped(vec, shape)

    from_P = np.sum(np.sum(P[:-1,:-1], axis=0)*np.arange(n_c))
    from_M = np.sum(P[-1,:]*M[-1,:,-1])+np.sum(P[:-1,-1]*M[:-1,-1,-1])
    return from_P+from_M