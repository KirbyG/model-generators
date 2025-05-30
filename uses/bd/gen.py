from generators.utils import Transition, DEC, INC, Dir

from generators.mean_field import build_mean_field_model
from generators.master_equation import build_master_equation_model
from generators.simulation import build_simulation_model
from generators.hybrid import build_hybrid_model

import numpy as np
from scipy.integrate import odeint
import pickle

μ, ν = 1, 0.05
T = 400
ts = np.linspace(0, T, 1000)
n_c = 8
N = 30

dimensions = ['pop']
shape = (n_c,)
transitions = [
    Transition({Dir('pop', INC)}, lambda n: μ*n),
    Transition({Dir('pop', DEC)}, lambda n: ν*n**2)
]

J_MF = build_mean_field_model(dimensions, transitions)
J_ME = build_master_equation_model(dimensions, (N,), transitions)
J_hybrid = build_hybrid_model(dimensions, shape, transitions)
J_sim = build_simulation_model(dimensions, transitions)

raw_MF = odeint(J_MF, [1], ts)[-1]

P0_ME = np.zeros(N)
P0_ME[1] = 1
raw_ME = odeint(J_ME, P0_ME, ts)[-1]

P0_hybrid = np.zeros(((n_c+1)*2))
P0_hybrid[1]=1
P0_hybrid[-1] = n_c
raw_hybrid = odeint(J_hybrid, P0_hybrid, ts)[-1]

raw_sim = []
for _ in range(1000):
    t = 0
    P_sim = [1.0]
    while t < T:
        dt, dP = J_sim(P_sim)
        t += dt
        P_sim += dP
    raw_sim.append(P_sim[0])

with open('uses/bd/cache/run.pickle', 'wb') as f:
    pickle.dump(
        {
            'MF': raw_MF,
            'ME': raw_ME,
            'hybrid': raw_hybrid,
            'sim': raw_sim,
            'n_c': n_c,
            'N': N
        },
        f
    )