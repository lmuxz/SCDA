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
model_type = "lgb"
version = "opt" # the version of prediction model
dim = 400

params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.04,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'num_threads': 30
}


# Train the model
for domain in ["books", "dvd", "elec", "kitchen"]:
    models = []
    perfs = []
    for seed in range(10):
        np.random.seed(seed)
        params['seed'] = seed

        print("Model", task, domain, flush=True)
        train, train_label, test, test_label = load_dataset("../data/", task, domain, data_type, dim)

        # Train the model with the best learning rate
        train, valid, train_label, valid_label = train_test_split(train, train_label, test_size=0.25, 
                                                                  shuffle=True, random_state=0)

        lgb_train = lgb.Dataset(train, train_label)
        lgb_valid = lgb.Dataset(valid, valid_label)

        model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=500, 
                        early_stopping_rounds=20, verbose_eval=None)

        models.append(model)
        pred = model.predict(test)
        perf = performance_logloss(pred, test_label)
        perfs.append(perf)

    model = models[np.argmax(perfs)]
    ## Save prediction model
    save_model(model, "../model/", task, domain, model_type, dim, version)

print("Done")