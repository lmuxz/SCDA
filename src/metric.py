import os
import numpy as np
import pandas as pd

import ot as pot
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from scipy.stats import entropy, rankdata

from sklearn.metrics import auc, precision_recall_curve, log_loss


def performance_pr_auc(pred, label, **kwargs):
    p, r, _ = precision_recall_curve(label, pred)
    order = np.argsort(r)
    return auc(r[order], p[order])

def performance_logloss(pred, label, pos_weight=1, **kwargs):
    pos_weight_list = np.ones(pred.shape[0])
    pos_weight_list[label==1] = pos_weight
    loss = - log_loss(label, pred, eps=1e-7, sample_weight=pos_weight_list, labels=[0, 1])
    return loss

def performance_acc(pred, label, threshold_acc=0.5, **kwargs):
    return ((pred > threshold_acc) == label).mean()

def of_uni_cate(source, target):
    source_counter = Counter(source.tolist())
    target_counter = Counter(target.tolist())

    modality = np.unique(np.r_[list(source_counter.keys()), list(target_counter.keys())])
    counts = np.zeros(len(modality))
    for i, m in enumerate(modality):
        counts[i] = source_counter.get(m, 0) + len(source) / len(target) * target_counter.get(m, 0)
    density = counts / counts.sum()

    res = 1 / (1+np.log(counts.sum()/counts).reshape((-1, 1)).dot(np.log(counts.sum()/counts).reshape(1, -1)))
    identity = modality.repeat(len(modality)).reshape((len(modality), len(modality))) == modality
    res[identity] = 1

    return res, modality

def similarity_to_dissimilarity(sim):
    max_sim = sim.max()
    return max_sim - sim

def wasserstein_1d(pred, pred_ref, **kwargs):
    return -pot.wasserstein_1d(pred, pred_ref)


def load_kaggle_logs(dir_path, metric, task, domain, model_type, period, version, exp, baseline=None, relative=False, verbose=True):
    file_path = os.path.join(dir_path, metric, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    n = len(exp)
    res = []
    b = None
    with open(file_path, "r") as f:
        for l in f:
            if exp in l:
                res.append(float(l[n:]))
            if baseline is not None:
                if baseline in l:
                    b = float(l[len(baseline):])
    res = np.array(res)
    if relative:
        res = (res - b) / abs(b) * 100
    if verbose:
        print("Mean: {:.4f}, Std: {:.4f}".format(np.mean(res), np.std(res)), flush=True)
    return res, b


def load_amazon_logs(dir_path, model_domain, model_type):
    path = os.path.join(dir_path, "amazon_{}_{}_400_opt".format(model_domain, model_type))
    
    target_domains = {}
    domain_list = []
    method_list = []
    with open(path, "r") as f:
        for l in f:
            if ":" in l:
                exp, perf = l.split(":")
                perf = float(perf)
                if ";" in exp:
                    method, domain = exp.split(";")
                    if method not in method_list:
                        method_list.append(method)
                        
                    if "source_" in domain:
                        target_domains.setdefault(domain, {})
                        target_domains[domain]["baseline"] = target_domains[model_domain]["baseline"]                        
                    else:
                        if "target_" in domain:
                            domain = domain[7:]
                        target_domains.setdefault(domain, {})
                        
                    target_domains[domain].setdefault(method, [])
                    target_domains[domain][method].append(perf)
                        
                    if domain not in domain_list:
                        domain_list.append(domain)
                else:
                    target_domains.setdefault(exp, {})
                    target_domains[exp]["baseline"] = perf
                    
                    if exp not in domain_list:
                        domain_list.append(exp)
                    
    return target_domains, domain_list, method_list

def analyse_amazon(model_type, metric):
    domain_enum = ["books", "dvd", "elec", "kitchen"]
    global_mean_df = None
    global_std_df = None

    for model_domain in domain_enum:
        logs, domain_list, method_list = load_amazon_logs("../logs/"+metric, model_domain, model_type)
        domain_list = [d for d in domain_enum if d != model_domain]
        mean_df = pd.DataFrame(index=method_list, columns=domain_list)
        std_df = pd.DataFrame(index=method_list, columns=domain_list)
        for domain in domain_list:
            domain_baseline = logs[domain]["baseline"]
            for method in method_list:
                perfs = logs[domain].get(method)
                if perfs:
                    perfs = (np.array(perfs) - domain_baseline) / abs(domain_baseline) * 100 #perfs = np.array(perfs)
                    mean_df.loc[method, domain] = np.mean(perfs)
                    std_df.loc[method, domain] = np.std(perfs)
        mean_df.columns = list(map(lambda x: model_domain+"_"+x, domain_list))
        std_df.columns = list(map(lambda x: model_domain+"_"+x, domain_list))

        if global_mean_df is None:
            global_mean_df = mean_df
            global_std_df = std_df
        else:
            global_mean_df = pd.concat([global_mean_df, mean_df], axis=1)
            global_std_df = pd.concat([global_std_df, std_df], axis=1)
            
    index_list = global_mean_df.index
    column_list = global_mean_df.columns

    res = global_mean_df.applymap(lambda x: '{0:.2f}'.format(x))+"+-"+global_std_df.applymap(lambda x: '{0:.2f}'.format(x))


    global_avg = global_mean_df.mean(axis=1)
    res["Avg"] = global_avg.map(lambda x: '{0:.2f}'.format(x))
    
    return res


def analyse_kaggle(model_type, metric, exp, sticks):
    task = "kaggle"
    period = [0, 1, 2, 3] # the period of data
    source_version = "opt" # the version of embedding matrix & prediction model that we use
    source_domain = "source"

    res = []
    baseline = []

    for e in exp:
        for p in period:
            r, b = load_kaggle_logs("../logs/", metric, task, source_domain, model_type, p, source_version, 
                                      e+": ", baseline="target: ", relative=True, verbose=False)
            res.append(r)
            baseline.append(b)

    cols = []
    df = pd.DataFrame({"Adaptation Method": sticks}) 
    cols.append("Adaptation Method")
    n = len(period)
    avg_total = []
    for i in range(n):
        avgs = list(map(np.mean, res[i::n]))
        avg_total.append(avgs)
        stds = list(map(np.std, res[i::n]))
        df["mean"+str(i)] = avgs
        df["std"+str(i)] = stds
        df["D-{} to M-{}".format(i+1, i+1)] = df[["mean"+str(i), "std"+str(i)]].apply(lambda x: "{:.2f}+-{:.2f}".format(x[0], x[1]), axis=1)
        cols.append("D-{} to M-{}".format(i+1, i+1))
    df["avg"] = np.array(avg_total).mean(axis=0)
    df["avg"] = df["avg"].apply(lambda x: "{:.2f}".format(x))
    cols.append("avg")

    return df[cols]