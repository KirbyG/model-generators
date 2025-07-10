# IMPORTS
from generators.utils import Transition, DEC, INC, Dir

from generators.mean_field import MeanFieldModel
from generators.master_equation import MasterEquationModel
from generators.simulation import SimModel
from generators.hybrid import HybridModel

import dill

# SETUP
K = 20
μ, ν, β = 0.04, 0.4, 0.06
n_c = 3
N = 20
T = 300

dimensions = ['fish', 'shark']
shape = (n_c, n_c)
transitions = [
    Transition({Dir('fish', INC)}, lambda f: μ*f*(K-f)),
    Transition({Dir('shark', DEC)}, lambda s: ν*s),
    Transition({Dir('fish', DEC), Dir('shark', INC)}, lambda f,s: β*f*s),
]

# MODELS
MF = MeanFieldModel(dimensions, None, transitions)
means = MF.empty_world()
means[0] = 12
means[1] = 8
MF.run(means, T)

ME = MasterEquationModel(dimensions, (N,N), transitions)
P0 = ME.empty_world()
P0[12, 8] = 1
ME.run(P0, T)

hybrid = HybridModel(dimensions, shape, transitions)
P, M = hybrid.empty_world()
P[n_c, n_c] = 1
M[n_c, :, :] = 3
M[:, n_c, :] = 3
M[n_c, n_c][0] = 12
M[n_c, n_c][1] = 8
hybrid.run((P, M), T)

sim = SimModel(dimensions, None, transitions)
values = sim.empty_world()
values[0] = 12
values[1] = 8
sim.run(values, T, trials=10)

# SAVE RESULTS
with open('uses/lv/cache/run.pickle', 'wb') as f:
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