import numpy as np
from generators.model import Model

class SimModel(Model):
    def __init__(self, dimensions, _, transitions, verbose=False):
        super().__init__(dimensions, _, transitions, verbose=verbose)
        def J(P):
            rates = [t.func(*np.extract(self.mask(t.dirs), P)) for t in transitions]
            R = np.sum(rates)
            if R == 0:
                return float('inf'), [0]*len(dimensions)
            dt = np.random.exponential(1/R)
            t = np.random.choice(transitions, p=rates/R)
            j = self.mask(t.dirs)
            return dt, j
        self.J = J
    
    def empty_world(self):
        return np.zeros(self.rank)
    
    def run(self, initial_conditions, T, trials=1000):
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
    
    def time_series(self, axis):
        ax = self.dimensions.index(axis)
        ts = []
        vs = []
        for trial in self.result:
            ts += list(trial[:, 0])
            vs += list(trial[:, ax+1])
        return ts, vs
    
    def distribution(self, axis):
        ax = self.dimensions.index(axis)
        vs = []
        for trial in self.result:
            vs.append(trial[-1][ax+1])
        counts = np.bincount(vs, minlength=3*int(max(vs)))
        print(vs)
        print(counts)
        return counts / len(vs)