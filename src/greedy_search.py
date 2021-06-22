# forward greedy search with feature cluster
import numpy as np
import gc

def get_perfs(model, valids, valid_label, particles, repeat, performance, **kwargs):
    n = valids[0].shape[0]

    data = []
    for particle in particles:
        data.append(np.hstack([valids[int(particle[i])][:,i].reshape(-1, 1) 
            for i in range(len(particle))]))

    perfs = []
    if len(data) > 0:
        data = np.vstack(data)
        preds = model.predict(data)
        del data
        gc.collect()

        for i in range(particles.shape[0]):
            pred = preds[i*n:(i+1)*n].reshape(repeat, -1).mean(axis=0)
            perf = performance(pred, valid_label, **kwargs)
            perfs.append(perf)
    
    return perfs


def forward_particle(particle, feature_cluster, feature_mask, n):
    particles = []
    for j in range(n):
        for i in range(len(feature_cluster)):
            p = particle.copy()
            add_n = 0
            for f in feature_cluster[i]:
                if particle[f] == 0 and f in feature_mask[j]:
                    add_n += 1
                    p[f] = j+1
            if add_n > 0:
                particles.append(p)
    return np.array(particles)


def forward_greedy_search(model, valids, valid_label, repeat, 
    performance, feature_cluster, best=None, feature_mask=None,
    verbose=True, **kwargs):
    if feature_mask is None:
        feature_mask = [np.arange(valids[0].shape[-1]) for i in range(len(valids)-1)]

    # init perf history and best history
    best_history = []
    evolution_perf = []

    # init best
    if best is None:
        best = np.zeros(valids[0].shape[-1])
    best_history.append(best)

    # init best_perf
    best_perf = get_perfs(model, valids, valid_label, best.reshape((1, -1)), repeat, performance, **kwargs)[0]
    evolution_perf.append(best_perf)
    if verbose:
        print(best_perf)

    forward_stop = False
    while not forward_stop:
        particles = forward_particle(best, feature_cluster, feature_mask, len(valids)-1)
        if particles.shape[0] > 0:
            perfs = get_perfs(model, valids, valid_label, particles, repeat, performance, **kwargs)
            i_best = np.argmax(perfs)

            forward_stop = perfs[i_best] < best_perf
        else:
            forward_stop = True

        if verbose:
            print("Performance:", perfs[i_best], flush=True)
            print("Selected Features:", np.where(particles[i_best]==1)[0], flush=True)

        # if not early stop
        if forward_stop:
            break
        # update best
        best = particles[i_best].copy()
        best_history.append(best)
        # update best_perf
        best_perf = perfs[i_best]
        evolution_perf.append(best_perf)

    return best, evolution_perf, best_history
