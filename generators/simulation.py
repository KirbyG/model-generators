import numpy as np
from generators.model import Model
from scipy.interpolate import make_smoothing_spline
from generators.utils import Transition
from typing import List

class SimModel(Model):
    def __init__(self, dimensions, _, transitions: List[Transition], verbose=False):
        """
        Create a model using the `next-reaction simulation method <https://en.wikipedia.org/wiki/Gillespie_algorithm>`_
        
        Parameters
        ----------
        dimensions : List[str]
            Names for the dimensions of the system.
        _ : any
            Shape parameter is unused for this model type.
        transitions : List[Transition]
            Specify rates and directions of model transitions.
        verbose : bool
            Indicate whether the model should log output.
        """
        super().__init__(dimensions, _, transitions, verbose=verbose)
        def J(P):
            rates = [t.func(*np.extract(t.dirs, P)) for t in transitions]
            R = np.sum(rates)
            if R == 0:
                return 1, [0]*len(dimensions)
            dt = np.random.exponential(1/R)
            t = np.random.choice(transitions, p=rates/R)
            j = t.dirs
            return dt, j
        self.J = J
    
    def empty_world(self):
        """Return a vector of zeroes with length equal to the rank of the system."""
        return np.zeros(self.rank)
    
    def run(self, initial_conditions, T, trials=1000):
        """Run _trials_ next-reaction simulations up to time T, starting from initial_conditions."""
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
    
    def time_series(self, axis):
        """View results in the given dimension as a time-series, averaged over all runs."""
        ax = self.dimensions.index(axis)

        splines = []
        for trial in self.result:
            irreg_t = list(trial[:, 0])
            irreg_v = list(trial[:, ax+1])
            splines.append(make_smoothing_spline(irreg_t, irreg_v))

        reg_ts = np.linspace(0, self.T, 4*self.T)
        reg_vs = np.empty(len(reg_ts))
        for i, t in enumerate(reg_ts):
            reg_vs[i] = np.average([spline(t) for spline in splines])
        return reg_vs
    
    def distribution(self, axis):
        """View results in the given dimension as a distribution over runs at time T."""
        ax = self.dimensions.index(axis)
        vs = []
        for trial in self.result:
            vs.append(trial[-1][ax+1])
        counts = np.bincount(vs, minlength=3*int(max(vs)))
        return counts / len(vs)
    
    def animation(self, axes):
        return super().animation(axes)