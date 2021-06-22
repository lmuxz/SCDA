import sys
sys.path.append("../src/")
sys.path.append("../model/")

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from nn_model import fully_connected_nn, fully_connected
from io_utils import load_dataset, save_model, model_log, load_model, load_pickle
from metric import performance_logloss, performance_acc

# Constant
task = "amazon" # the dataset that we are working on
data_type = "msda" # the type of data that we are dealing with
version = "opt" # the version of prediction model
dim = 400

model_type = "lgb"


# Logloss
for model_domain in ["books", "dvd", "elec", "kitchen"]:
    model = load_model("../model/", task, model_domain, model_type, dim, version)
    for data_domain in ["books", "dvd", "elec", "kitchen"]:
        train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, dim)

        pred = model.predict(test)

        perf = performance_logloss(pred, test_label)
        model_log("../logs/logloss", task, model_domain, model_type, dim, version, "{}: {}".format(data_domain, perf))


# acc
for model_domain in ["books", "dvd", "elec", "kitchen"]:
    model = load_model("../model/", task, model_domain, model_type, dim, version)
    for data_domain in ["books", "dvd", "elec", "kitchen"]:
        train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, dim)

        pred = model.predict(test)

        perf = performance_acc(pred, test_label)
        model_log("../logs/acc", task, model_domain, model_type, dim, version, "{}: {}".format(data_domain, perf))


model_type = "nn"

# Logloss
for model_domain in ["books", "dvd", "elec", "kitchen"]:
    model = load_model("../model/", task, model_domain, model_type, dim, version)
    for data_domain in ["books", "dvd", "elec", "kitchen"]:
        train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, dim)

        pred = model.predict(test)

        perf = performance_logloss(pred, test_label)
        model_log("../logs/logloss", task, model_domain, model_type, dim, version, "{}: {}".format(data_domain, perf))


# acc
for model_domain in ["books", "dvd", "elec", "kitchen"]:
    model = load_model("../model/", task, model_domain, model_type, dim, version)
    for data_domain in ["books", "dvd", "elec", "kitchen"]:
        train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, dim)

        pred = model.predict(test)

        perf = performance_acc(pred, test_label)
        model_log("../logs/acc", task, model_domain, model_type, dim, version, "{}: {}".format(data_domain, perf))
        
print("Done")