import numpy as np
from scipy.integrate import odeint
from generators.utils import powerset, Transition, DEC, Po
from generators.model import Model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HybridModel(Model):
    def to_shaped(self, states: np.ndarray):

        P = np.reshape(states[:self.num_states], self.shape)

        M = np.reshape(states[self.num_states:], self.extended_shape)

        return P, M

    def to_linear(self, P: np.ndarray, M: np.ndarray):
        return np.concatenate((P.flatten(), M.flatten()))
    
    def __init__(self, dimensions, critical_points, base_transitions, verbose=False):
        self.critical_points = np.array(critical_points)
        super().__init__(dimensions, self.critical_points+1, base_transitions, verbose=verbose)
        self.extended_shape = np.append(self.shape+1, [self.rank])
        self.num_states = np.prod(self.shape)

        # generate virtual transitions
        transitions = base_transitions.copy()
        for t in base_transitions:
            for subset in powerset(t.dirs_list):
                transitions.append(Transition(
                    t.dirs_list-subset,
                    t.func,
                    t.dimensions,
                    virtual=True,
                    fails=subset
                ))

        def J(states, _):
            # reshape states into joint prob and means
            P, M = self.to_shaped(states)

            JP = np.zeros(self.shape)

            JM = np.zeros(self.extended_shape)

            # update means according to mean field equations
            with np.nditer(M, flags=['multi_index']) as it:
                for _ in it:
                    cell = np.array(it.multi_index[:-1])
                    dim = it.multi_index[-1]
                    critical_dims = cell==self.critical_points
                    if np.any(critical_dims) and critical_dims[dim]:
                        state = np.where(critical_dims, M[*cell], cell)
                        JM[*cell][dim] = np.sum([
                            t.dirs[dim]*t.func(
                                *np.extract(t.dirs, state)
                            ) for t in transitions if not t.virtual
                        ])

            # update probs according to master equations
            with np.nditer(P, flags=['multi_index']) as it:
                for _ in it:
                    cell = np.array(it.multi_index)
                    for t in transitions:
                        j_in = 0
                        log = ''

                        origin = cell - t.dirs
                        critical_origins = self.critical_points==origin
                        critical_cells = self.critical_points==cell
                        critical_exits = critical_origins & ~critical_cells
                        critical_stays = critical_origins & critical_cells
                        log += f'##########f={" ".join(f"{dimensions[i]}={c}" for i,c in enumerate(cell))}:t={t}##########\n'

                        # virtual transitions must originate in critical dimensions that they fail to escape
                        if t.virtual and np.any(critical_stays!=t.fails.astype(bool)):
                            log += 'virtual skip\n'
                            continue

                        # skip transitions originating above the bounds of the system
                        if np.any(self.shape==origin):
                            continue
                        
                        # skip transitions originating below 0
                        if np.any(origin==-1):
                            continue
                            
                        state = np.where(critical_stays, M[*origin], origin)
                        j_in = P[*origin] * t.func(*np.extract(t.dirs+t.fails, state))
                        log += f'j_in(base):{j_in}\n'
                        # need to successfully exit critical dimensions
                        j_in *= np.prod([
                            Po(n_c, mean_value) for n_c, mean_value in zip(
                                np.extract(critical_exits, self.critical_points),
                                np.extract(critical_exits, M[*origin])
                            )
                        ])
                        log += f'j_in(critical exit roll successes)(red):{j_in}\n'
                        # virtual transitions caused by failed exits need to actually fail all exits
                        j_in *= np.prod([
                            1-Po(n_c, mean_value) for n_c, mean_value in zip(
                                np.extract(t.fails==DEC, self.critical_points),
                                np.extract(t.fails==DEC, M[*origin])
                            )
                        ])
                        log += f'j_in(virtual: exit roll fails)(green):{j_in}\n'

                        if verbose and j_in > 0:
                            print(log)
                        JP[*cell] += j_in
                        JP[*origin] -= j_in


            return self.to_linear(JP, JM)
        self.J = J

    def empty_world(self):
        return self.to_shaped(np.zeros(self.num_states + np.prod(self.extended_shape)))
    
    def run(self, initial_conditions, T):
        self.T = T
        linear = odeint(self.J, self.to_linear(*initial_conditions), np.linspace(0, T, 4*T))
        self.result = [self.to_shaped(world) for world in linear]
    
    def extract(self, P, M, ax):
        p_slices = [slice(None)] * self.rank 
        p_slices[ax] = slice(-1)
        p = np.sum(P[tuple(p_slices)], axis=tuple(set(range(self.rank))-{ax}))

        m_slices = [slice(None)] * self.rank
        m_slices[ax] = slice(-1, None)
        m = zip(
            P[tuple(m_slices)].flatten(),
            M[*tuple(m_slices), ax].flatten()
        )

        return p, m

    def mean(self, P, M, ax):
        p, m = self.extract(P, M, ax)

        from_p = np.sum(p * range(self.critical_points[ax]))
        # print(f'p:{from_p}')
        # print(list(m))
        from_m = np.sum([t[0] * t[1] for t in m])
        # print(f'm:{from_m}')

        return from_p + from_m

    def time_series(self, axis):
        ax = self.dimensions.index(axis)
        return [self.mean(P, M, ax) for P, M in self.result]
    
    def distribution(self, axis):
        ax = self.dimensions.index(axis)
        n_c = self.shape[ax]
        P, M = self.result[-1]

        p, m = self.extract(P, M, ax)


        peak = self.mean(P, M, ax)

        arr = np.array([np.array([Po(n, mean) for n in range(int(n_c), 3*int(peak))])*prob for prob, mean in m])
        from_m = np.sum(
            arr,
            axis=0
        )

        return np.concatenate((p, from_m))
    
    def animation(self, axes):
        cmap = plt.cm.get_cmap('viridis')
        fig, ax = plt.subplots()
        # plt.imshow(np.log10(data + epsilon), cmap='viridis')
        img = ax.imshow(np.log10(self.result[0][0]+1e-10), cmap=cmap, animated=True)

        def update(frame):
            img.set_array(np.log10(self.result[frame][0]+1e-10))
            return [img]

        anim  = FuncAnimation(fig, update, frames=self.T+1)
        return anim