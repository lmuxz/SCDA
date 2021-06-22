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

from nn_model import fully_connected_embed, embed_nn
from io_utils import load_dataset, load_model, model_log
from metric import performance_logloss, performance_pr_auc
from train_utils import extend_dataset, reduce_dataset, sample_validation_data

from labelshift_correction import build_pivot_dataset, adjust_model


# ### Setting

# In[ ]:


model_type = "dan"

task = "kaggle" # the dataset that we are working on
data_type = "cate" # the type of data that we are dealing with
num_dim = 43
epoch = 25
batch_size = 1024
period = [0, 1, 2, 3] # the period of data
cate_index = 8 # the index of the last categorical feature
version = "opt" # the version of embedding matrix & prediction model
source_domain = "source"
target_domain = "target"

device = torch.device("cuda") # device of training 
        
embedding_input = [3, 131, 4, 483, 103, 5, 106, 4] # different levels of categorical features
embedding_dim = [1, 3, 1, 4, 3, 1, 3, 1] # embedding dimension

ratio = 0.2


# ### Hyper-parameter selection

# In[ ]:


beta_range = [0.005, 0.01, 0.1]
period_beta = []

for p in period:
    perf_beta = []
    for beta in beta_range:
        perfs = []
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)

            source_train, source_train_label, source_test, source_test_label = load_dataset("../data/", 
                                                                                            task, source_domain, data_type, p)
            target_train, target_train_label, target_test, target_test_label = load_dataset("../data/", 
                                                                                            task, target_domain, data_type, p)

            # get target_factor and source_factor
            target_factor = (target_train_label[:, 1]==0).sum() / target_train_label[:, 1].sum()
            source_factor = (source_train_label[:, 1]==0).sum() / source_train_label[:, 1].sum()

            # adjusting source train dataset
            source_train, source_train_label, source_index = build_pivot_dataset(
                source_train, source_train_label[:,1], target_factor, source_factor)
            
            # train undersample
            valid_index, source_train_label = sample_validation_data(task, source_train_label, ratio)
            source_train = source_train[valid_index]
            
            valid_index, target_train_label = sample_validation_data(task, target_train_label, ratio)
            target_train = target_train[valid_index]

            source_train, source_valid, source_train_label, source_valid_label = train_test_split(
                source_train, source_train_label, test_size=0.25, shuffle=False, random_state=0)

            embed = embed_nn(embedding_input, embedding_dim, num_dim)
            model = fully_connected_embed(embed, cate_index, device)

            source_index, target_index = reduce_dataset(source_train, target_train)

            model.fit(source_train[source_index], source_train_label[source_index], 
                      target_train[target_index],
                      source_valid, source_valid_label, 
                      epoch=epoch, batch_size=batch_size, lr=0.005, beta=beta,
                      early_stop=False, verbose=False)

            pred = model.predict(source_test)
            perf = performance_logloss(pred, source_test_label[:, 1])
            perfs.append(perf)
        perf_beta.append(np.mean(perfs))
    period_beta.append(beta_range[np.argmax(perf_beta)])

path = os.path.join("./results", task, model_type, 
                     "{}_{}".format(model_type, version))
if not os.path.exists(path):
    os.makedirs(path)
np.save(os.path.join(path, "period_beta"), period_beta)

print("Optimal beta for each period:", period_beta, flush=True)


# ### Adaptation

# In[ ]:


for p in period:
    beta = period_beta[p]
    for seed in range(10):
        print("Period", p, "Seed", seed, flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        source_train, source_train_label, source_test, source_test_label = load_dataset("../data/", 
                                                                                        task, source_domain, data_type, p)
        target_train, target_train_label, target_test, target_test_label = load_dataset("../data/", 
                                                                                        task, target_domain, data_type, p)

        # get target_factor and source_factor
        target_factor = (target_train_label[:, 1]==0).sum() / target_train_label[:, 1].sum()
        source_factor = (source_train_label[:, 1]==0).sum() / source_train_label[:, 1].sum()

        # adjusting source train dataset
        source_train, source_train_label, source_index = build_pivot_dataset(
            source_train, source_train_label[:,1], target_factor, source_factor)

        source_train, source_valid, source_train_label, source_valid_label = train_test_split(
            source_train, source_train_label, test_size=0.25, shuffle=False, random_state=0)

        embed = embed_nn(embedding_input, embedding_dim, num_dim)
        model = fully_connected_embed(embed, cate_index, device)

        source_index, target_index = reduce_dataset(source_train, target_train)

        model.fit(source_train[source_index], source_train_label[source_index, 1], 
                  target_train[target_index],
                  source_valid, source_valid_label[:, 1], 
                  epoch=epoch, batch_size=batch_size, lr=0.005, beta=beta,
                  early_stop=False, verbose=False)
        
        path = os.path.join("./results", task, model_type, 
                     "{}_{}".format(model_type, version), 
                     "period{}".format(p), "exp{}".format(seed))
        if not os.path.exists(path):
            os.makedirs(path)


        # source prediction
        pred = model.predict(source_test)
        np.save(os.path.join(path, "source_test_pred"), pred.astype(np.float16))

        perf = performance_logloss(pred, source_test_label[:, 1])
        print("Source Prediction logloss:", perf, flush=True)
        model_log("../logs/logloss", task, source_domain, "nn", p, version, 
                     "source_{}: {}".format(model_type, perf))

        perf = performance_pr_auc(pred, source_test_label[:, 1])
        print("Source Prediction pr_auc:", perf, flush=True)
        model_log("../logs/pr_auc", task, source_domain, "nn", p, version, 
                     "source_{}: {}".format(model_type, perf))


        # target prediction
        pred = model.predict(target_test)
        np.save(os.path.join(path, "target_test_pred"), pred.astype(np.float16))

        perf = performance_logloss(pred, target_test_label[:, 1])
        print("Target Prediction logloss:", perf, flush=True)
        model_log("../logs/logloss", task, source_domain, "nn", p, version, 
                     "target_{}: {}".format(model_type, perf))

        perf = performance_pr_auc(pred, target_test_label[:, 1])
        print("Target Prediction pr_auc:", perf, flush=True)
        model_log("../logs/pr_auc", task, source_domain, "nn", p, version, 
                     "target_{}: {}".format(model_type, perf))


# In[ ]:




