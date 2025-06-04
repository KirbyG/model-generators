import matplotlib.pyplot as plt
import numpy as np
import dill

MF, ME, hybrid, sim, T = tuple([None]*5)
with open('uses/lv/cache/run.pickle', 'rb') as f:
    locals().update(dill.load(f))

fig, (series, dist) = plt.subplots(1, 2)
legend = ['Mean-Field', 'Master-Equation', 'Hybrid', 'Simulation']
plt.title('-- fish / .. shark')

ts = np.linspace(0, T, 4*T)
series.plot(ts, MF.time_series('fish'), color='blue')
series.plot(ts, MF.time_series('fish'), color='blue')
series.plot(ts, MF.time_series('fish'), color='blue')
series.plot(ts, MF.time_series('fish'), color='blue')