from generators.utils import Transition, DEC, INC, Dir, to_linear, to_shaped

from generators.mean_field import build_mean_field_model
from generators.master_equation import build_master_equation_model
from generators.simulation import build_simulation_model
from generators.hybrid import build_hybrid_model

import numpy as np
from scipy.integrate import odeint
import pickle

K = 20
μ, ν, β = 0.04, 0.4, 0.06
n_c = 3
N = 20
T = 100
ts = np.linspace(0, T, 200)

dimensions = ['fish', 'shark']
shape = (n_c, n_c)
transitions = [
    Transition({Dir('fish', INC)}, lambda f: μ*f*(K-f)),
    Transition({Dir('shark', DEC)}, lambda s: ν*s),
    Transition({Dir('fish', DEC), Dir('shark', INC)}, lambda f,s: β*f*s),
]

J_MF = build_mean_field_model(dimensions, transitions)
J_ME = build_master_equation_model(dimensions, (N,N), transitions)
J_hybrid = build_hybrid_model(dimensions, shape, transitions)
J_sim = build_simulation_model(dimensions, transitions)


raw_MF = odeint(build_mean_field_model(dimensions, transitions), [12, 8], ts)

P0_ME = np.zeros((N,N))
P0_ME[12, 8] = 1
P0_ME = np.reshape(P0_ME, (N**2))
raw_ME = odeint(J_ME, P0_ME, ts)

vec0 = np.zeros(((1+2)*(n_c+1)**2))
P0, M0 = to_shaped(vec0, shape)
P0[-1,-1] = 1
M0[-1,:,0] = 3
M0[:,-1,1] = 3
M0[-1,-1,0] = 12
M0[-1,-1,1] = 8
vec0 = to_linear((P0, M0), shape)
raw_hybrid = odeint(J_hybrid, vec0, ts)

raw_sim = []
for _ in range(1000):
    t = 0
    P_sim = [12.0, 8.0]
    print(f'#########{_}#########')
    while t < T:
        dt, dP = J_sim(P_sim)
        t += dt
        P_sim += dP
    raw_sim.append(P_sim[1])

with open('uses/lv/cache/run.pickle', 'wb') as f:
    pickle.dump(
        {
            'MF': raw_MF,
            'ME': raw_ME,
            'hybrid': raw_hybrid,
            'sim': raw_sim,
            'n_c': n_c,
            'N': N
        }
    )

# TODO the models should encapsulate result processing logic