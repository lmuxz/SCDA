#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

sys.path.append("../src/")
sys.path.append("../model/")


# In[ ]:


import numpy as np
import torch

from datetime import datetime

from io_utils import load_dataset, load_model, model_log
from metric import performance_logloss, performance_pr_auc

from coordinate_ot_adaptation import adaptation
from labelshift_correction import build_pivot_dataset, adjust_model
from greedy_search import forward_greedy_search
from train_utils import sample_validation_data

from sklearn.model_selection import train_test_split


# ### Setting

# In[ ]:


model_type = "nn" # for NN models
# model_type = "lgb" # for GBDT models

# number of threads
njobs = 20


# ### Adaptation

# In[ ]:


source_version = "opt" # the version of embedding matrix & prediction model that we use
task = "kaggle"
data_type = "cate"
num_dim = 43
period = [0, 1, 2, 3]
cate_index = 8
source_domain = "source"
target_domain = "target"
ratio = 0.20
version = "supervised_feature_selection"

num_selected_features = [[] for _ in period]

for seed in range(10):
    for p in period:
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Period:", p, seed, flush=True)
        model = load_model("../model/", task, source_domain, model_type, p, source_version)
        source_train, source_train_label, source_test, source_test_label = load_dataset("../data/", 
                                                                                        task, source_domain, data_type, p)
        target_train, target_train_label, target_test, target_test_label = load_dataset("../data/", 
                                                                                        task, target_domain, data_type, p)

        # get target_factor and source_factor
        target_factor = (target_train_label[:, 1]==0).sum() / target_train_label[:, 1].sum()
        source_factor = (source_train_label[:, 1]==0).sum() / source_train_label[:, 1].sum()

        # adjusting the classifier
        model = adjust_model(model, target_factor, source_factor)

        # adjusting source train dataset
        source_train, source_train_label, source_index = build_pivot_dataset(
            source_train, source_train_label[:,1], target_factor, source_factor)

        # source and target datat undersampling
        source_train_index, _ = sample_validation_data(task, source_train_label, ratio)
        source_train = source_train[source_train_index]
        
        target_train_index, target_train_label = sample_validation_data(task, target_train_label, ratio)
        target_train = target_train[target_train_index]
        
        # source train prediction basedline
        pred_source = model.predict(source_train)

        # init adaptation & fit & transform
        adapt = adaptation(cate_dim=cate_index, num_dim=num_dim)
        adapt.fit(target_train, source_train, lmbda=1e-1)

        target_train_trans = adapt.transform(target_train, repeat=5, njobs=njobs)

        params = {
            "model": model, 
            "valids": [np.tile(target_train, (5, 1)), target_train_trans],
            "valid_label": target_train_label, 
            "repeat": 5,
            "performance": performance_logloss,
            "feature_cluster":[[i] for i in range(target_train.shape[-1])],
            "best": None,
            "feature_mask": None,
            "verbose": False
        }

        path = os.path.join("./results", task, version, 
                     "{}_{}".format(model_type, source_version), 
                     "period{}".format(p), "exp{}".format(seed))
        if not os.path.exists(path):
            os.makedirs(path)

        # greedy feature selection
        feature_mask, evolution_perf, best_history = forward_greedy_search(**params)
        np.save(os.path.join(path, "feature_mask"), feature_mask)
        num_selected_features[p].append(feature_mask.sum())

        # target test transformation based on selected features
        target_test_trans = adapt.transform(target_test, repeat=20, interpolation=feature_mask, njobs=njobs)

        pred = model.predict(target_test_trans).reshape(20, -1).mean(axis=0)
        np.save(os.path.join(path, "target_test_pred"), pred)

        perf = performance_pr_auc(pred, target_test_label[:, 1])
        model_log("../logs/pr_auc/", task, source_domain, model_type, p, source_version, 
                 "{}: {}".format(version, perf))
        print("Target Prediction pr_auc", perf, flush=True)

        perf = performance_logloss(pred, target_test_label[:, 1])
        model_log("../logs/logloss/", task, source_domain, model_type, p, source_version, 
                 "{}: {}".format(version, perf))
        print("Target Prediction logloss", perf, flush=True)

num_selected_features = np.array(num_selected_features)
path = os.path.join("./results", task, version, 
                     "{}_{}".format(model_type, source_version))
np.save(os.path.join(path, "num_selected_features"), num_selected_features)

print("Number of Selected Feature:", num_selected_features.mean(axis=1))
print("Std of Selected Feature:", num_selected_features.std(axis=1))


# In[ ]:


# print("Number of Selected Feature:", num_selected_features.mean(axis=1))
# print("Std of Selected Feature:", num_selected_features.std(axis=1))

