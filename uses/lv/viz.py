import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('uses/lv/cache/run.pickle', 'rb') as f:
    locals().update(pickle.load(f))

result_MF_2D = MF[:,1][-1]
result_ME_2D = [np.sum(np.reshape(state, (N,N)), axis=0) for state in ME][-1]
#TODO get shark distribution instead of mean for this viz
result_hybrid_2D = [extract_mean(state, 1, shape) for state in raw_hybrid_2D][-1]
counts = np.bincount(raw_sim_2D, minlength=N)
result_sim_2D = counts / len(raw_sim_2D)

plt.axvline(x=result_MF_2D, color='r', linestyle='--', linewidth=2)
plt.plot(range(N), result_ME_2D)
plt.plot(range(N), result_hybrid_2D)
plt.plot(range(N), result_sim_2D)
plt.xlabel('Sharks')
plt.ylabel('Probability')
plt.legend(['Mean-Field', 'Master-Equation', 'Hybrid', 'Simulation'])
plt.show()