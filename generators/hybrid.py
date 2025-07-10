import numpy as np
from scipy.integrate import odeint
from generators.utils import powerset, Transition, DEC, Po
from generators.model import Model

class HybridModel(Model):
    def to_shaped(self, states: np.ndarray):

        P = np.reshape(states[:self.num_states], self.full_shape)

        M = np.reshape(states[self.num_states:], self.extended_shape)

        return P, M

    def to_linear(self, P: np.ndarray, M: np.ndarray):
        return np.concatenate((P.flatten(), M.flatten()))
    
    def __init__(self, dimensions, shape, base_transitions, verbose=False):
        super().__init__(dimensions, shape, base_transitions, verbose=verbose)
        self.full_shape = tuple(np.array(self.shape)+1)
        self.extended_shape = tuple(np.append(np.array(self.shape)+1, [self.rank]))
        self.num_states = np.prod(self.full_shape)

        # generate virtual transitions
        transitions = base_transitions.copy()
        for t in base_transitions:
            for subset in powerset(t.dirs):
                transitions.append(Transition(
                    t.dirs-subset,
                    t.func,
                    virtual=True,
                    fails=subset
                ))

        def J(states, _):
            # reshape states into joint prob and means
            P, M = self.to_shaped(states)

            P = np.pad(
                P,
                [(0,1)]*self.rank,
                mode='constant',
                constant_values=0
            )
            JP = np.zeros(np.array(shape)+1)

            extended_shape = np.append(np.array(shape)+1, [self.rank])
            JM = np.zeros(extended_shape)

            # update means according to mean field equations
            with np.nditer(M, flags=['multi_index']) as it:
                for m in it:
                    cell = it.multi_index[:-1]
                    if np.any(np.array(shape)==np.array(cell)):
                        dim = it.multi_index[-1]
                        critical_dims = np.array(shape)-np.array(cell)==0
                        if np.any(critical_dims) and critical_dims[dim]:
                            JM[cell][dim] = np.sum([
                                self.mask(t.dirs)[dim]*t.func(
                                    *np.extract(self.mask(t.dirs), M[cell])
                                ) for t in transitions if not t.virtual
                            ])

            # update probs according to master equations
            with np.nditer(P, flags=['multi_index']) as it:
                for p in it:
                    cell = it.multi_index
                    if np.any(np.array(shape)+1==np.array(cell)):
                        continue
                    for t in transitions:
                        j_in = 0
                        log = ''
                        # incoming
                        origin = np.array(cell) - self.mask(t.dirs)
                        critical_origins = np.array(shape)-origin==0
                        critical_cells = np.array(shape)-np.array(cell)==0
                        critical_exits = critical_origins & ~critical_cells
                        critical_stays = critical_origins & critical_cells
                        log += f'##########f={" ".join(f"{dimensions[i]}={c}" for i,c in enumerate(cell))}:t={t}##########\n'

                        #colors
                        # virtual transitions originate in critical dimensions that they fail to escape
                        if t.virtual and np.any(critical_stays!=self.mask(t.fails).astype(bool)):
                            log += 'virtual skip\n'
                            continue

                        if not np.any(np.array(shape)+1==np.array(origin)):
                            j_in = P[tuple(origin)] * t.func(*np.extract(
                                self.mask(t.dirs)+self.mask(t.fails),
                                np.where(critical_stays, M[tuple(origin)], origin)
                            ))
                            log += f'j_in(base):{j_in}\n'
                            # all transitions
                            j_in *= np.prod([
                                Po(n_c, mean_value) for n_c, mean_value in zip(
                                    np.extract(critical_exits, shape),
                                    np.extract(critical_exits, M[tuple(origin)])
                                )
                            ])
                            log += f'j_in(critical exit roll successes)(red):{j_in}\n'
                            # virtual transition exit rolls
                            j_in *= np.prod([
                                1-Po(n_c, mean_value) for n_c, mean_value in zip(
                                    np.extract(self.mask(t.fails)==DEC, shape),
                                    np.extract(self.mask(t.fails)==DEC, M[cell])
                                )
                            ])
                            log += f'j_in(virtual: exit roll fails)(green):{j_in}\n'

                            if verbose and j_in > 0:
                                print(log)
                            JP[cell] += j_in
                            JP[tuple(origin)] -= j_in


            return self.to_linear(JP, JM)
        self.J = J

    def empty_world(self):
        return self.to_shaped(np.zeros(self.num_states + np.prod(self.extended_shape)))
    
    def run(self, initial_conditions, T):
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

        from_p = np.sum(p * range(self.shape[ax]))
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
        # print(len(P))
        # print(len(M))
        p, m = self.extract(P, M, ax)
        # print(len(p))
        # print(len(list(m)))
        # print(list(m)[0])

        peak = self.mean(P, M, ax)
        # print(peak)
        # print(range(int(n_c), 3*int(peak)))
        # print(np.array([Po(n, peak) for n in range(int(n_c), 3*int(peak))]))
        # print([prob for prob in list(m)])
        arr = np.array([np.array([Po(n, mean) for n in range(int(n_c), 3*int(peak))])*prob for prob, mean in m])
        # print(arr)
        from_m = np.sum(
            arr,
            axis=0
        )
        # print(len(p))
        # print(len(from_m))
        return np.concatenate((p, from_m))