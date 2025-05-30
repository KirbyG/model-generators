from generators.utils import make_mask
import numpy as np
def build_simulation_model(dimensions, transitions):
    mask = make_mask(dimensions)
    def J(P):
        rates = [t.func(*np.extract(mask(t.dirs), P)) for t in transitions]
        R = np.sum(rates)
        if R == 0:
            return float('inf'), [0]*len(dimensions)
        dt = np.random.exponential(1/R)
        t = np.random.choice(transitions, p=rates/R)
        j = mask(t.dirs)
        return dt, j
    return J