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
    feature_cluster, bias_tol_range, best=None, feature_mask=None, n_bootstrap=200,
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
            preds = get_predictions(model, valids, particles, repeat)
            preds_all = np.vstack([best_prediction, preds])
            # print("get_predictions done", flush=True)

            # print("get_quantile start", flush=True)
            # get quantile for preds
            quantiles = [get_quantile(pred) for pred in preds]
            # get quantile for previous prediction
            quantile_previous = get_quantile(best_prediction)
            # print("get_quantile done", flush=True)

            # print("get_hts start", flush=True)
            # get hts
            hts = np.array([get_ht(quantile, valid_label) for quantile in quantiles])
            # get previous_ht
            ht_previous = get_ht(quantile_previous, valid_label)
            ht_all = np.vstack([ht_previous, hts])
            # print("get_hts done", flush=True)

            hts_pred_diff = []
            for i in range(ht_all.shape[0]):
                hts_pred_diff.append(abs(preds_all - ht_all[i]))
            hts_pred_diff = np.array(hts_pred_diff)
            
            # Find the optimal delta
            ht_diff = ht_all.max(axis=0) - ht_all.min(axis=0)
            if ht_tol is None:
                dis = []
                bias = []
                bias_stable_masks = []
                for bias_tol in bias_tol_range:
                    pred_mask = get_prediction_stable_points(ht_previous, hts, bias_tol)
                    masks = [pred_mask]
                    
                    if instance_mask is not None:
                        masks.append(instance_mask)
                    bias_stable_mask = np.all(masks, axis=0)
                    bias_stable_masks.append(bias_stable_mask)

                    valid_stable = valid[bias_stable_mask]
                    
                    train_data = np.vstack([valid, valid_stable])
                    train_label = np.r_[np.ones(valid.shape[0]), np.zeros(valid_stable.shape[0])]

                    params = {
                        'boosting': 'gbdt',
                        'objective': 'binary',
                        'metric': 'binary',
                        'learning_rate': 0.04,
                        'max_depth': 4,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'num_threads': 30,
                        'pos_weight': bias_stable_mask.sum() / len(bias_stable_mask),
                        'seed': 0,
                        'verbose': -1,
                    }
                    
                    train, v_data, train_label, v_label = train_test_split(train_data, train_label, test_size=0.25, shuffle=True)
                    lgb_train = lgb.Dataset(train, train_label, categorical_feature=range(cate_index))
                    lgb_valid = lgb.Dataset(v_data, v_label, categorical_feature=range(cate_index))
                    
                    clf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=5000, 
                                    early_stopping_rounds=20, verbose_eval=False)
                    

                    # err = (1-clf.predict(valid)).mean() + clf.predict(valid_stable).mean()
                    err = (1-(clf.predict(valid)>0.5)).mean() + (clf.predict(valid_stable)>0.5).mean()
                    dis.append(1-err)
                    bias.append(ht_diff[bias_stable_mask].mean())

                # return np.array(dis), np.array(bias), bias_tol_range
                bias_min_ind = np.argmin(np.array(dis) + np.array(bias))
                ht_tol = bias_tol_range[bias_min_ind]
                if verbose:
                    print("Best Tolerance:", ht_tol, flush=True)
                instance_mask = bias_stable_masks[bias_min_ind]
            else:
                pred_mask = get_prediction_stable_points(ht_previous, hts, ht_tol)
                masks = [pred_mask]
                if instance_mask is not None:
                    masks.append(instance_mask)
                instance_mask = np.all(masks, axis=0)

            instance_index = np.where(instance_mask==1)[0]
            if len(instance_index) == 0:
                break

            # bootstrap instance_index
            bootstrap_index = [
                np.random.choice(instance_index, len(instance_index), replace=True) for _ in range(n_bootstrap)
            ]
            candidates = np.zeros(len(particles))
            # print("bootstrap start", flush=True)
            for ind in bootstrap_index:
                # print("bootstrap undergoing", flush=True)
                hts_pred_diff_example = hts_pred_diff[:, :, ind].mean(axis=2).T
                i_best = np.argmin(hts_pred_diff_example.min(axis=1))

                if i_best != 0:
                    candidates[i_best-1] += 1
            # print("bootstrap done", flush=True)

            i_best = np.argmax(candidates)

            forward_stop = candidates[i_best]/n_bootstrap < 0.50
        else:
            forward_stop = True
        

        if verbose:
            print("Instance Percentage: {:.4f}".format(instance_mask.mean()), "Votes Percentage: {:.2f}".format(candidates[i_best]/n_bootstrap), flush=True)
            print("Selected Features:", np.where(particles[i_best]==1)[0], flush=True)

        # if not early stop
        if forward_stop:
            break
        # update best
        best_prediction = preds[i_best]
        best = particles[i_best].copy()
        best_history.append(best)

    return best, best_history
