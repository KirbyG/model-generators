import numpy as np
from scipy.integrate import odeint
from generators.model import Model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class MasterEquationModel(Model):
    def __init__(self, dimensions, shape, transitions, verbose=False):
        super().__init__(dimensions, shape, transitions, verbose=verbose)
        def J(states, _):
            P = np.reshape(states, self.shape)
            j = np.zeros(self.shape)
            with np.nditer(P, flags=['multi_index']) as it:
                for _ in it:
                    cell = np.array(it.multi_index)
                    for t in transitions:
                        origin = cell - t.dirs
                        if np.any(shape==origin):
                            continue
                        if np.any(origin==-1):
                            continue
                        dj = P[*origin] * t.func(*np.extract(t.dirs, origin))
                        if verbose:
                            print(f't={t}:{origin}--[{dj}]-->{cell}')
                        j[*cell] += dj
                        j[*origin] -= dj
            return j.flatten()
        self.J = J
    
    def empty_world(self):
        return np.zeros(self.shape)
    
    def run(self, P0: np.ndarray, T):
        self.T = T
        ts = np.linspace(0, T, 4*T)
        self.result = np.reshape(
            odeint(self.J, P0.flatten(), ts),
            (len(ts), *self.shape)
        )

    def time_series(self, axis):
        ax = self.dimensions.index(axis)
        N = self.shape[ax]
        dists = np.sum(self.result, axis=tuple(np.array(list((set(range(self.rank))-{ax})))+1))
        support = np.reshape(range(N), (1,N))
        return np.sum(dists*support, axis=1)
    
    def distribution(self, axis):
        ax = self.dimensions.index(axis)
        return np.sum(self.result[-1], axis=tuple((set(range(self.rank))-{ax})))
    
    def animation(self, axes):
        cmap = plt.cm.get_cmap('viridis')
        fig, ax = plt.subplots()
        # plt.imshow(np.log10(data + epsilon), cmap='viridis')
        img = ax.imshow(np.log10(self.result[0, :, :]+1e-10), cmap=cmap, animated=True)

        def update(frame):
            img.set_array(np.log10(self.result[frame, :, :]+1e-10))
            return [img]

        anim  = FuncAnimation(fig, update, frames=self.T+1)
        return anim