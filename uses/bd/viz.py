import matplotlib.pyplot as plt
import numpy as np
import dill

MF, ME, hybrid, sim, T = tuple([None]*5)
with open('uses/bd/cache/run.pickle', 'rb') as f:
    locals().update(dill.load(f))

fig, (ts, dist) = plt.subplots(1, 2)
legend = ['Mean-Field', 'Master-Equation', 'Hybrid', 'Simulation']

ts.plot(np.linspace(0, T, 4*T), MF.time_series('pop'), color='blue')
ts.plot(np.linspace(0, T, 4*T), ME.time_series('pop'), color='orange')
ts.plot(np.linspace(0, T, 4*T), hybrid.time_series('pop'), color='green')
ts.scatter(*sim.time_series('pop'), color='red')

ts.set_xlabel('Time')
ts.set_ylabel('Mean Pop')

N = ME.shape[0]
dist.axvline(x=MF.distribution('pop'), color='blue')
dist.plot(range(N), ME.distribution('pop')[:N], color='orange')
dist.plot(range(N), hybrid.distribution('pop')[:N], color='green')
dist.plot(range(N), sim.distribution('pop')[:N], color='red')

dist.set_xlabel('Pop')
dist.set_ylabel('Prob')
fig.legend(legend)
plt.show()

# QUESTIONS FOR LHD
# asymmetric transitions
# plotting sim results
# known high-dim systems
# validate LV against mean-flame paper backend
# what is going on with the rho term?