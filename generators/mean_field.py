from generators.utils import make_mask
import numpy as np
def build_mean_field_model(dimensions, transitions):
    mask = make_mask(dimensions)
    def J(states, _):
        j = np.zeros(len(states))
        for t in transitions:
            j += mask(t.dirs)*t.func(*np.extract(mask(t.dirs), states))
        return j
    return J