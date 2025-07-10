import numpy as np
from generators.model import Model
from scipy.interpolate import interp1d

class SimModel(Model):
    def __init__(self, dimensions, _, transitions, verbose=False):
        super().__init__(dimensions, _, transitions, verbose=verbose)
        def J(P):
            rates = [t.func(*np.extract(self.mask(t.dirs), P)) for t in transitions]
            R = np.sum(rates)
            if R == 0:
                return 1, [0]*len(dimensions)
            dt = np.random.exponential(1/R)
            t = np.random.choice(transitions, p=rates/R)
            j = self.mask(t.dirs)
            return dt, j
        self.J = J
    
    def empty_world(self):
        return np.zeros(self.rank)
    
    def run(self, initial_conditions, T, trials=1000):
        self.T = T
        result = []
        for _ in range(trials):
            t = 0
            vec = np.copy(initial_conditions)
            trial = np.concatenate(([t], vec))
            while t < T:
                dt, dP = self.J(vec)
                t += dt
                vec += dP
                trial = np.vstack((trial, np.concatenate(([t], vec))))
            result.append(trial)
        self.result = result #List of time series of vector
    
    def time_series(self, axis): # broken
        ax = self.dimensions.index(axis)
        
        interps = []
        for trial in self.result:
            irreg_t = list(trial[:, 0])
            irreg_v = list(trial[:, ax+1])
            interps.append(interp1d(irreg_t, irreg_v))

        reg_ts = np.arange(0, self.T+1, 1)
        reg_vs = np.empty(self.T+1)
        for t in reg_ts:
            reg_vs[t] = np.average([interp(t) for interp in interps])
        return reg_ts, reg_vs
    
    def distribution(self, axis): # works
        ax = self.dimensions.index(axis)
        vs = []
        for trial in self.result:
            vs.append(trial[-1][ax+1])
        counts = np.bincount(vs, minlength=3*int(max(vs)))
        return counts / len(vs)