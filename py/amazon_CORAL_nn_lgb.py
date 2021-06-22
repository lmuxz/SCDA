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

from io_utils import load_dataset, load_model, model_log
from metric import performance_logloss, performance_acc

from labelshift_correction import build_pivot_dataset, adjust_model
from train_utils import sample_validation_data
from CORAL import CORAL


# ### Amazon

# In[ ]:


version = "coral"

source_version = "opt" # the version of embedding matrix & prediction model that we use

task = "amazon"
data_type = "msda"
dim = 400


# ### Adaptation

# In[ ]:


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
                
                # fit numerical data
                coral = CORAL()
                coral.fit(target_train, source_train)
                
                # transform target_test_num
                target_test_trans = coral.transform(target_test)

                for model_type in ["nn", "lgb"]:
                    model = load_model("../model/", task, model_domain, model_type, dim, source_version)
                    
                    # prediction and save log
                    path = os.path.join("./results", task, version, 
                                 "{}_{}".format(model_type, source_version), 
                                 model_domain, data_domain, "exp{}".format(seed))
                    if not os.path.exists(path):
                        os.makedirs(path)

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


# In[ ]:




