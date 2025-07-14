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
T = 200
F0, S0 = 4, 5

dimensions = ['fish', 'shark']
critical_points = (n_c, n_c)
transitions = [
    Transition({Dir('fish', INC)}, lambda f: μ*f*(K-f), dimensions),
    Transition({Dir('shark', DEC)}, lambda s: ν*s, dimensions),
    Transition({Dir('fish', DEC), Dir('shark', INC)}, lambda f,s: β*f*s, dimensions),
]

# MODELS
MF = MeanFieldModel(dimensions, None, transitions)
means = MF.empty_world()
means[0] = F0
means[1] = S0
MF.run(means, T)

ME = MasterEquationModel(dimensions, (N,N), transitions)
P0 = ME.empty_world()
P0[F0, S0] = 1
ME.run(P0, T)

hybrid = HybridModel(dimensions, critical_points, transitions)
P, M = hybrid.empty_world()
P[n_c, n_c] = 1
M[n_c, :, :] = 3
M[:, n_c, :] = 3
M[n_c, n_c][0] = F0
M[n_c, n_c][1] = S0
hybrid.run((P, M), T)

sim = SimModel(dimensions, None, transitions)
values = sim.empty_world()
values[0] = F0
values[1] = S0
sim.run(values, T, trials=1000)

# SAVE RESULTS
with open('uses/lv/cache/run4-5.pickle', 'wb') as f:
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