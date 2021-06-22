import sys
sys.path.append("../src/")
sys.path.append("../model/")

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from nn_model import fully_connected_nn, fully_connected
from io_utils import load_dataset, save_model, model_log, load_model, load_pickle
from metric import performance_logloss, performance_acc

# Constant
task = "amazon" # the dataset that we are working on
data_type = "msda" # the type of data that we are dealing with
model_type = "nn"
version = "opt" # the version of prediction model
dim = 400

epoch = 50
batch_size = 128

device = torch.device("cuda") # device of training


# Train the model
for domain in ["books", "dvd", "elec", "kitchen"]:
    models = []
    perfs = []
    for seed in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Model", task, domain, flush=True)
        train, train_label, test, test_label = load_dataset("../data/", task, domain, data_type, dim)

        # Train the model with the best learning rate
        train, valid, train_label, valid_label = train_test_split(train, train_label, test_size=0.25, 
                                                                  shuffle=True, random_state=0)
        dnn = fully_connected_nn(dim)
        model = fully_connected(dnn, device)

        model.fit(train, train_label, 
                  train, 
                  valid, valid_label, 
                  epoch=epoch, batch_size=batch_size, lr=0.01, beta=0, 
                  early_stop=False, verbose=False)

        models.append(model)
        pred = model.predict(test)
        perf = performance_logloss(pred, test_label)
        perfs.append(perf)

    model = models[np.argmax(perfs)]
    ## Save prediction model
    save_model(model, "../model/", task, domain, model_type, dim, version)

print("Done")