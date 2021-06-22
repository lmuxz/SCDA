import sys
sys.path.append("../src/")
sys.path.append("../model/")

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold

from io_utils import load_dataset, save_model, model_log, load_model, load_pickle
from metric import performance_pr_auc, performance_logloss

# Constant
task = "kaggle" # the dataset that we are working on
data_type = "cate" # the type of data that we are dealing with
period = [0, 1, 2, 3] # the period of data
version = "opt" # the version of embedding matrix & prediction model

model_type = "lgb"

# Logloss
for p in period:
    for model_domain in ["source"]:
        model = load_model("../model/", task, model_domain, model_type, p, version)
        for data_domain in ["source", "target"]:
            train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, p)

            pred = model.predict(test)

            perf = performance_logloss(pred, test_label[:, 1])
            model_log("../logs/logloss", task, model_domain, model_type, p, version, "{}: {}".format(data_domain, perf))

# PR_AUC
for p in period:
    for model_domain in ["source"]:
        model = load_model("../model/", task, model_domain, model_type, p, version)
        for data_domain in ["source", "target"]:
            train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, p)

            pred = model.predict(test)

            perf = performance_pr_auc(pred, test_label[:, 1])   
            model_log("../logs/pr_auc", task, model_domain, model_type, p, version, "{}: {}".format(data_domain, perf))


model_type = "nn"
# Logloss
for p in period:
    for model_domain in ["source"]:
        model = load_model("../model/", task, model_domain, model_type, p, version)
        for data_domain in ["source", "target"]:
            train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, p)

            pred = model.predict(test)

            perf = performance_logloss(pred, test_label[:, 1])
            model_log("../logs/logloss", task, model_domain, model_type, p, version, "{}: {}".format(data_domain, perf))

# PR_AUC
for p in period:
    for model_domain in ["source"]:
        model = load_model("../model/", task, model_domain, model_type, p, version)
        for data_domain in ["source", "target"]:
            train, train_label, test, test_label = load_dataset("../data/", task, data_domain, data_type, p)

            pred = model.predict(test)

            perf = performance_pr_auc(pred, test_label[:, 1])   
            model_log("../logs/pr_auc", task, model_domain, model_type, p, version, "{}: {}".format(data_domain, perf))


print("Done")