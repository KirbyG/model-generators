import matplotlib.pyplot as plt
import numpy as np
import dill

MF, ME, hybrid, sim, T = tuple([None]*5)
with open('uses/lv/cache/run.pickle', 'rb') as f:
    locals().update(dill.load(f))

fig, (series, dist) = plt.subplots(1, 2)
legend = [
    'Fish Mean-Field',
    'Fish Master-Equation',
    'Fish Hybrid',
    'Fish Sim',
    'Shark Mean-Field',
    'Shark Master-Equation',
    'Shark Hybrid',
    'Shark Sim'
]
ts = np.linspace(0, T, 4*T)

print(hybrid.time_series('fish'))

series.plot(ts, MF.time_series('fish'), color='blue', linestyle='dashed')
series.plot(ts, ME.time_series('fish'), color='orange', linestyle='dashed')
series.plot(ts, hybrid.time_series('fish'), color='green', linestyle='dashed')
series.scatter(*sim.time_series('fish'), color='red')

series.plot(ts, MF.time_series('shark'), color='blue')
series.plot(ts, ME.time_series('shark'), color='orange')
series.plot(ts, hybrid.time_series('shark'), color='green')
series.scatter(*sim.time_series('shark'), color='red')


series.set_xlabel('Time')
series.set_ylabel('Mean Pop')



fig.legend(legend)
plt.show()