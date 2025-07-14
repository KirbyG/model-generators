# IMPORTS
from generators.utils import Transition, DEC, INC, Dir

from generators.mean_field import MeanFieldModel
from generators.master_equation import MasterEquationModel
from generators.simulation import SimModel
from generators.hybrid import HybridModel

import dill

# SETUP
μ, ν = 1, 0.05
T = 50
n_c = 16
N = 30

dimensions = ['pop']
shape = (n_c,)
transitions = [
    Transition({Dir('pop', INC)}, lambda n: μ*n, dimensions),
    Transition({Dir('pop', DEC)}, lambda n: ν*n**2, dimensions)
]

# MODELS
MF = MeanFieldModel(dimensions, None, transitions)
means = MF.empty_world()
means[0] = 1
MF.run(means, T)

ME = MasterEquationModel(dimensions, (N,), transitions)
P0 = ME.empty_world()
P0[1] = 1
ME.run(P0, T)

hybrid = HybridModel(dimensions, shape, transitions)
P, M = hybrid.empty_world()
P[1] = 1
M[-1] = n_c
hybrid.run((P, M), T)

sim = SimModel(dimensions, None, transitions)
values = sim.empty_world()
values[0] = 1
sim.run(values, T)

# SAVE RESULTS
with open('uses/bd/cache/run16.pickle', 'wb') as f:
    dill.dump(
        {
            'MF': MF,
            'ME': ME,
            'hybrid': hybrid,
            'sim': sim,
            'T': T
        },
        f
    )