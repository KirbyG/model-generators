import matplotlib.pyplot as plt
import numpy as np
from generators.utils import Po
import pickle

MF, ME, hybrid, sim, n_c, N = tuple([None]*6)
with open('uses/bd/cache/run.pickle', 'rb') as f:
    locals().update(pickle.load(f))

result_hybrid = np.concat((hybrid[:n_c], [Po(n,hybrid[-1]) for n in range(n_c, N)]))

counts = np.bincount(sim, minlength=N)[:N]
result_sim = counts / len(sim)

plt.axvline(x=MF, color='r', linestyle='--', linewidth=2)
plt.plot(range(N), ME)
plt.plot(range(N), result_hybrid)
plt.plot(range(N), result_sim)
plt.xlabel('Population')
plt.ylabel('Probability')
plt.legend(['Mean-Field', 'Master-Equation', 'Hybrid', 'Simulation'])
plt.show()