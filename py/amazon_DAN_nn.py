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

from sklearn.model_selection import train_test_split, KFold

from nn_model import fully_connected_nn, fully_connected
from io_utils import load_dataset, load_model, model_log
from metric import performance_logloss, performance_acc
from train_utils import extend_dataset, reduce_dataset


# ### Setting

# In[ ]:


model_type = "dan"

task = "amazon"
data_type = "msda"

dim = 400
epoch = 50
batch_size = 128
version = "opt" # the version of embedding matrix & prediction model

device = torch.device("cuda") # device of training 


# ### Beta selection

# In[ ]:


beta_range = [0.01, 0.05, 0.1]
model_beta = {}
for model_domain in ["books", "dvd", "elec", "kitchen"]:
    model_beta[model_domain] = {}
    for data_domain in ["books", "dvd", "elec", "kitchen"]:
        if data_domain != model_domain:
            perfs_beta = []
            for beta in beta_range:
                perfs = []
                for seed in range(10):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # Load dataset
                    source_train, source_train_label, source_test, source_test_label = load_dataset("../data/", 
                                                                                            task, model_domain, data_type, dim)
                    target_train, target_train_label, target_test, target_test_label = load_dataset("../data/", 
                                                                                            task, data_domain, data_type, dim)

                    # Split train valid data
                    source_train, source_valid, source_train_label, source_valid_label = train_test_split(
                        source_train, source_train_label, test_size=0.25, shuffle=True, random_state=0)

                    # init model & train
                    dnn = fully_connected_nn(dim)
                    model = fully_connected(dnn, device)

                    source_index, target_index = reduce_dataset(source_train, target_train)

                    model.fit(source_train[source_index], source_train_label[source_index], 
                              target_train[target_index],
                              source_valid, source_valid_label, 
                              epoch=epoch, batch_size=batch_size, lr=0.005, beta=beta,
                              early_stop=False, verbose=False)

                    # predict on source test
                    pred = model.predict(source_test)
                    perf = performance_logloss(pred, source_test_label)
                    perfs.append(perf)
                perfs_beta.append(np.mean(perfs))
            model_beta[model_domain][data_domain] = beta_range[np.argmax(perfs_beta)]
            
path = os.path.join("./results", task, model_type, 
                     "{}_{}".format(model_type, version))
if not os.path.exists(path):
    os.makedirs(path)
np.save(os.path.join(path, "model_beta"), model_beta)

print("Optimal beta for each period:", model_beta, flush=True)


# ### Adaptation

# In[ ]:


for seed in range(10):
    for model_domain in ["books", "dvd", "elec", "kitchen"]:
        for data_domain in ["books", "dvd", "elec", "kitchen"]:
            if data_domain != model_domain:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Load dataset
                source_train, source_train_label, source_test, source_test_label = load_dataset("../data/", 
                                                                                        task, model_domain, data_type, dim)
                target_train, target_train_label, target_test, target_test_label = load_dataset("../data/", 
                                                                                        task, data_domain, data_type, dim)

                # Split train valid data
                source_train, source_valid, source_train_label, source_valid_label = train_test_split(
                    source_train, source_train_label, test_size=0.25, shuffle=True, random_state=0)

                # init model & train
                dnn = fully_connected_nn(dim)
                model = fully_connected(dnn, device)

                source_index, target_index = reduce_dataset(source_train, target_train)

                model.fit(source_train[source_index], source_train_label[source_index], 
                          target_train[target_index],
                          source_valid, source_valid_label, 
                          epoch=epoch, batch_size=batch_size, lr=0.005, beta=model_beta[model_domain][data_domain],
                          early_stop=False, verbose=False)

                # prediction and save log
                path = os.path.join("./results", task, model_type, 
                             "{}_{}".format(model_type, version), 
                             model_domain, data_domain, "exp{}".format(seed))
                if not os.path.exists(path):
                    os.makedirs(path)

                # Source prediction
                pred = model.predict(source_test)
                np.save(os.path.join(path, "source_test_pred"), pred.astype(np.float16))

                perf = performance_logloss(pred, source_test_label)
                model_log("../logs/logloss/", task, model_domain, "nn", dim, version, 
                         "{};source_{}: {}".format(model_type, data_domain, perf))
                print("Source Prediction logloss", model_domain, data_domain, perf, flush=True)

                perf = performance_acc(pred, source_test_label)
                model_log("../logs/acc/", task, model_domain, "nn", dim, version, 
                         "{};source_{}: {}".format(model_type, data_domain, perf))
                print("Source Prediction accuracy", model_domain, data_domain, perf, flush=True)
                
                # Traget prediction
                pred = model.predict(target_test)
                np.save(os.path.join(path, "target_test_pred"), pred.astype(np.float16))

                perf = performance_logloss(pred, target_test_label)
                model_log("../logs/logloss/", task, model_domain, "nn", dim, version, 
                         "{};target_{}: {}".format(model_type, data_domain, perf))
                print("Target Prediction logloss", model_domain, data_domain, perf, flush=True)

                perf = performance_acc(pred, target_test_label)
                model_log("../logs/acc/", task, model_domain, "nn", dim, version, 
                         "{};target_{}: {}".format(model_type, data_domain, perf))
                print("Target Prediction accuracy", model_domain, data_domain, perf, flush=True)


# In[ ]:




