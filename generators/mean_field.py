import numpy as np
from scipy.integrate import odeint
from generators.model import Model

class MeanFieldModel(Model):
    def __init__(self, dimensions, _, transitions, verbose=False):
        super().__init__(dimensions, _, transitions, verbose=verbose)
        def J(states, _):
            j = np.zeros(len(states))
            for t in transitions:
                j += t.dirs*t.func(*np.extract(t.dirs, states))
            return j
        self.J = J

    def empty_world(self):
        return np.zeros(self.rank)
    
    def run(self, initial_condition, T):
        self.result = odeint(self.J, initial_condition, np.linspace(0, T, 4*T))
    
    def time_series(self, axis):
        ax = self.dimensions.index(axis)
        return self.result[:, ax]
    
    def distribution(self, axis):
        ax = self.dimensions.index(axis)
        mean = self.result[-1, ax]
        return mean