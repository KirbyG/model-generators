from generators.utils import make_mask, powerset, to_linear, to_shaped, Transition, DEC, Po
import numpy as np
def build_hybrid_model(dimensions, shape, base_transitions, verbose=False):
    num_states = np.prod(np.array(shape)+1)
    rank = len(shape)

    mask = make_mask(dimensions)

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
        P, M = to_shaped(states, shape)

        P = np.pad(
            P,
            [(0,1)]*rank,
            mode='constant',
            constant_values=0
        )
        JP = np.zeros(np.array(shape)+1)

        extended_shape = np.append(np.array(shape)+1, [rank])
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
                            mask(t.dirs)[dim]*t.func(
                                *np.extract(mask(t.dirs), M[cell])
                            ) for t in transitions if not t.virtual
                        ])

        # update probs according to master equations
        with np.nditer(P, flags=['multi_index']) as it:
            for p in it:
                cell = it.multi_index
                if np.any(np.array(shape)+1==np.array(cell)):
                    continue
                for t in transitions:
                    j_in, j_out = 0, 0
                    log = ''
                    # incoming
                    origin = np.array(cell) - mask(t.dirs)
                    critical_origins = np.array(shape)-origin==0
                    critical_cells = np.array(shape)-np.array(cell)==0
                    critical_exits = critical_origins & ~critical_cells
                    critical_stays = critical_origins & critical_cells
                    log += f'##########f={" ".join(f"{dimensions[i]}={c}" for i,c in enumerate(cell))}:t={t}##########\n'
                    # virtual transitions originate in critical dimensions that they fail to escape
                    if t.virtual and np.any(critical_stays!=mask(t.fails).astype(bool)):
                        log += 'virtual skip\n'
                        continue

                    if not np.any(np.array(shape)+1==np.array(origin)):
                        j_in = P[tuple(origin)] * t.func(*np.extract(
                            mask(t.dirs)+mask(t.fails),
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
                                np.extract(mask(t.fails)==DEC, shape),
                                np.extract(mask(t.fails)==DEC, M[cell])
                            )
                        ])
                        log += f'j_in(virtual: exit roll fails)(green):{j_in}\n'

                        if verbose and j_in > 0:
                            print(log)
                        JP[cell] += j_in
                        JP[tuple(origin)] -= j_in


        return to_linear((JP, JM), shape)
    return J