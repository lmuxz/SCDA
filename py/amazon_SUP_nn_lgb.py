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
from metric import performance_logloss, performance_acc

from coordinate_ot_adaptation import adaptation
from labelshift_correction import build_pivot_dataset, adjust_model
from greedy_search import forward_greedy_search
from train_utils import sample_validation_data


# ### Setting

# In[ ]:


version = "SUP"
source_version = "opt" # the version of embedding matrix & prediction model that we use

task = "amazon"
data_type = "msda"
dim = 400


# ### Adaptation

# In[ ]:


num_selected_features = {"lgb":{}, "nn":{}}
for seed in range(10):
    for model_domain in ["books", "dvd", "elec", "kitchen"]:
        for data_domain in ["books", "dvd", "elec", "kitchen"]:
            if data_domain != model_domain:
                torch.manual_seed(seed)
                np.random.seed(seed)

                source_train, source_train_label, source_test, source_test_label = load_dataset("../data/", 
                                                                                        task, model_domain, data_type, dim)
                target_train, target_train_label, target_test, target_test_label = load_dataset("../data/", 
                                                                                        task, data_domain, data_type, dim)
                
                adapt = adaptation(cate_dim=0, num_dim=dim)
                adapt.fit(target_train, source_train)
                
                target_train_trans = adapt.transform(target_train, repeat=1, njobs=20)

                for model_type in ["lgb", "nn"]:
                    num_selected_features[model_type].setdefault(model_domain, {})
                    num_selected_features[model_type][model_domain].setdefault(data_domain, [])
                    model = load_model("../model/", task, model_domain, model_type, dim, source_version)
                    
                    params = {
                        "model": model, 
                        "valids": [target_train, target_train_trans],
                        "valid_label": target_train_label, 
                        "repeat": 1,
                        "performance": performance_logloss,
                        "feature_cluster":[[i] for i in range(target_train.shape[-1])],
                        "best": None,
                        "feature_mask": None,
                        "verbose": False
                    }

                    # greedy feature selection
                    feature_mask, evolution_perf, best_history = forward_greedy_search(**params)
                    
                    path = os.path.join("./results", task, version, 
                                 "{}_{}".format(model_type, source_version), 
                                 model_domain, data_domain, "exp{}".format(seed))
                    if not os.path.exists(path):
                        os.makedirs(path)
                        
                    np.save(os.path.join(path, "feature_mask"), feature_mask)
                    num_selected_features[model_type][model_domain][data_domain].append(feature_mask.sum())
                    
                    # target test transformation based on selected features
                    target_test_trans = adapt.transform(target_test, repeat=1, interpolation=feature_mask, njobs=20)
                    
                    pred = model.predict(target_test_trans)
                    np.save(os.path.join(path, "target_test_pred"), pred)
                    
                    perf = performance_logloss(pred, target_test_label)
                    model_log("../logs/logloss/", task, model_domain, model_type, dim, source_version, 
                             "{};{}: {}".format(version, data_domain, perf))
                    print("Prediction logloss", model_domain, data_domain, perf, flush=True)

                    perf = performance_acc(pred, target_test_label)
                    model_log("../logs/acc/", task, model_domain, model_type, dim, source_version, 
                             "{};{}: {}".format(version, data_domain, perf))
                    print("Prediction accuracy", model_domain, data_domain, perf, flush=True)


path = os.path.join("./results", task, version)
np.save(os.path.join(path, "num_selected_features"), num_selected_features)


# In[ ]:


for model_domain in ["books", "dvd", "elec", "kitchen"]:
    for data_domain in ["books", "dvd", "elec", "kitchen"]:
        if model_domain != data_domain:
            for model_type in ["nn", "lgb"]:
                print(model_domain, data_domain, model_type, "avg num features:",
                      np.mean(num_selected_features[model_type][model_domain][data_domain]))
                print(model_domain, data_domain, model_type, "std num features:",
                      np.std(num_selected_features[model_type][model_domain][data_domain]))


# In[ ]:




