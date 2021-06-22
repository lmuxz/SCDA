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
model_type = "lgb"
period = [0, 1, 2, 3] # the period of data
cate_index = 8 # the index of the last categorical feature
version = "opt" # the version of embedding matrix & prediction model


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

for domain in ["source"]:
    for p in period:
        models = []
        perfs = []
        for seed in range(10):
            np.random.seed(seed)
            params["seed"] = seed
            
            print("Model", task, domain, p, flush=True)
            train, train_label, test, test_label = load_dataset("../data/", task, domain, data_type, p)

            ## Train the model with the best learning rate
            train, valid, train_label, valid_label = train_test_split(train, train_label, test_size=0.25, 
                                                                      shuffle=False, random_state=0)

            lgb_train = lgb.Dataset(train, train_label[:,1], categorical_feature=range(cate_index))
            lgb_valid = lgb.Dataset(valid, valid_label[:,1], categorical_feature=range(cate_index))

            model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=5000, 
                            early_stopping_rounds=20, verbose_eval=None)
            ###
            
            models.append(model)
            pred = model.predict(test)
            perf = performance_logloss(pred, test_label[:, 1])
            perfs.append(perf)

        model = models[np.argmax(perfs)]
        ## Save prediction model
        save_model(model, "../model/", task, domain, model_type, p, version)

print("Done")