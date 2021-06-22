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
from prediction_stable_feature_selection import scda
from train_utils import sample_validation_data


# ### Setting

# In[ ]:


model_type = "nn" # for NN models
# model_type = "lgb" # for GBDT models

# number of threads
njobs = 20


# ### Adaptation

# In[ ]:


source_version = "opt"
task = "kaggle"
data_type = "cate"
num_dim = 43
period = [0, 1, 2, 3]
cate_index = 8
source_domain = "source"
target_domain = "target"
version = "all_adaptation"


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

        # init adaptation & fit & transform
        adapt = adaptation(cate_dim=cate_index, num_dim=num_dim)
        adapt.fit(target_train, source_train, lmbda=1e-1)
        
        # target test transformation based on selected features
        target_test_trans = adapt.transform(target_test, repeat=20, njobs=njobs)
            
        pred = model.predict(target_test_trans).reshape(20, -1).mean(axis=0)
        
        path = os.path.join("./results", task, version, 
                     "{}_{}".format(model_type, source_version), 
                     "period{}".format(p), "exp{}".format(seed))
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "target_test_pred"), pred)

        perf = performance_pr_auc(pred, target_test_label[:, 1])
        model_log("../logs/pr_auc/", task, source_domain, model_type, p, source_version, 
                 "{}: {}".format(version, perf))
        print("Target Prediction pr_auc", perf, flush=True)

        perf = performance_logloss(pred, target_test_label[:, 1])
        model_log("../logs/logloss/", task, source_domain, model_type, p, source_version, 
                 "{}: {}".format(version, perf))
        print("Target Prediction logloss", perf, flush=True)


# In[ ]:




