import sys
sys.path.append("../src/")
sys.path.append("../model/")

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from nn_model import fully_connected_embed, embed_nn
from io_utils import load_dataset, save_model, model_log, load_model, load_pickle
from metric import performance_pr_auc, performance_logloss

# Constant
task = "kaggle" # the dataset that we are working on
data_type = "cate" # the type of data that we are dealing with
model_type = "nn"
version = "opt" # the version of prediction model
num_dim = 43

epoch = 25
batch_size = 1024
period = [0, 1, 2, 3] # the period of data
cate_index = 8 # the index of the last categorical feature

device = torch.device("cuda") # device of training 
        
embedding_input = [3, 131, 4, 483, 103, 5, 106, 4] # different levels of categorical features
embedding_dim = [1, 3, 1, 4, 3, 1, 3, 1] # embedding dimension

# Train the model
for domain in ["source"]:
    for p in period:
        models = []
        perfs = []
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)

            print("Model", task, domain, p, flush=True)
            train, train_label, test, test_label = load_dataset("../data/", task, domain, data_type, p)

            # Train the model with the best learning rate
            train, valid, train_label, valid_label = train_test_split(train, train_label, test_size=0.25, 
                                                                      shuffle=False, random_state=0)
            embed = embed_nn(embedding_input, embedding_dim, num_dim)
            model = fully_connected_embed(embed, cate_index, device)

            model.fit(train, train_label[:, 1], 
                      train, 
                      valid, valid_label[:, 1], 
                      epoch=epoch, batch_size=batch_size, lr=0.005, beta=0, 
                      early_stop=False, verbose=False)
            
            models.append(model)
            pred = model.predict(test)
            perf = performance_logloss(pred, test_label[:, 1])
            perfs.append(perf)

        model = models[np.argmax(perfs)]
        ## Save prediction model
        save_model(model, "../model/", task, domain, model_type, p, version)

print("Done")