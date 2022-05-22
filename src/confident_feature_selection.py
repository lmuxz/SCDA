# forward greedy search with feature cluster
import numpy as np
import lightgbm as lgb
import gc

from scipy.stats import entropy, rankdata

from sklearn.model_selection import train_test_split


def get_quantile(pred):
    return rankdata(pred, "average") / len(pred)


def get_prediction_stable_points(pred_previous, preds, tol=0.01):
    preds = preds + [pred_previous]
    masks = []
    for i in range(len(preds)):
        for j in range(len(preds)):
            masks.append(abs(preds[j] - preds[i]) <= tol)
    stable_points_mask = np.all(masks, axis=0)
    return stable_points_mask


def get_ht(quantile, pred_ref):
    return np.quantile(pred_ref, quantile)


def get_predictions(model, valids, particles, repeat):
    n = valids[0].shape[0]

    data = []
    for particle in particles:
        data.append(np.hstack([valids[int(particle[i])][:,i].reshape(-1, 1) 
            for i in range(len(particle))]))

    predictions = []
    if len(data) > 0:
        data = np.vstack(data)
        preds = model.predict(data)
        del data
        gc.collect()

        for i in range(particles.shape[0]):
            pred = preds[i*n:(i+1)*n].reshape(repeat, -1).mean(axis=0)
            predictions.append(pred)
    
    return np.array(predictions)


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


def scda(model, valid, valid_trans, valid_label, cate_index, repeat, 
    feature_cluster, best=None, feature_mask=None, n_bootstrap=200,
    verbose=True):
    valids = [np.tile(valid, (repeat, 1)), valid_trans]

    if feature_mask is None:
        feature_mask = [np.arange(valids[0].shape[-1]) for i in range(len(valids)-1)]

    # init perf history and best history
    best_history = []

    # init best
    if best is None:
        best = np.zeros(valids[0].shape[-1])

    # init best_perf
    best_prediction = get_predictions(model, valids, best.reshape((1, -1)), repeat)[0]
    instance_mask = None

    forward_stop = False
    ht_tol = None
    while not forward_stop:
        particles = forward_particle(best, feature_cluster, feature_mask, len(valids)-1)
        if particles.shape[0] > 0:
            # print("get_predictions start", flush=True)

            neg_threshold = np.quantile(best_prediction, 0.05)
            pos_threshold = np.quantile(best_prediction, 0.99)

            instance_ind = np.hstack([np.where(best_prediction >= pos_threshold)[0], 
                np.where(best_prediction <= neg_threshold)[0]])

            if verbose:
                print("Length:", len(instance_ind))

            preds = get_predictions(model, valids, particles, repeat)

            preds_diff = abs(preds[:, instance_ind] - best_prediction[instance_ind])

            # bootstrap instance_index
            bootstrap_index = [
                np.random.choice(preds_diff.shape[-1], preds_diff.shape[-1], replace=True) for _ in range(n_bootstrap)
            ]
            candidates = np.zeros(len(particles))
            # print("bootstrap start", flush=True)
            for ind in bootstrap_index:
                i_best = np.argmin(preds_diff[:, ind].mean(axis=1))
                candidates[i_best] += 1
            # print("bootstrap done", flush=True)

            i_best = np.argmax(candidates)

            forward_stop = candidates[i_best]/n_bootstrap < 0.50
        else:
            forward_stop = True
        

        if verbose:
            print("Votes Percentage: {:.2f}".format(candidates[i_best]/n_bootstrap), flush=True)
            if particles.shape[0] > 0:
                print("Selected Features:", np.where(particles[i_best]==1)[0], flush=True)

        # if not early stop
        if forward_stop:
            break
        # update best
        best_prediction = preds[i_best]
        best = particles[i_best].copy()
        best_history.append(best)

    return best, best_history
