from generators.utils import make_mask
import numpy as np
def build_master_equation_model(dimensions, shape, transitions, verbose=False):
    mask = make_mask(dimensions)
    def J(states, _):
        P = np.reshape(states, shape)
        j = np.zeros(shape)
        with np.nditer(P, flags=['multi_index']) as it:
            for _ in it:
                cell = it.multi_index
                for t in transitions:
                    origin = np.array(cell) - mask(t.dirs)
                    if np.any(np.array(shape)==origin):
                        continue
                    if np.any(origin==-1):
                        continue
                    dj = P[tuple(origin)] * t.func(*np.extract(mask(t.dirs), origin))
                    if verbose:
                        print(f't={t}:{origin}--[{dj}]-->{cell}')
                    j[cell] += dj
                    j[tuple(origin)] -= dj
        return j.flatten()
    return J