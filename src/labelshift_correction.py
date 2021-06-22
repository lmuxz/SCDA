# Ajust label shift by introducing the pivot domain s'
import numpy as np
from sklearn.model_selection import train_test_split


def build_pivot_dataset(source, source_label, target_factor, source_factor=None):
    """
    Inputs:
        source_factor: ratio n_non_fraud / n_fraud
        target_factor: ratio n_non_fraud / n_fraud
    """
    non_fraud_n = (source_label==0).sum()
    if source_factor is None:
        source_factor = non_fraud_n / source_label.sum()
    
    ratio = target_factor / source_factor
    if ratio > 1:
        replace = True
    else:
        replace = False
    
    fraud_index = np.where(source_label==1)[0]
    genuine_index = np.where(source_label==0)[0]

    genuine_index = np.random.choice(genuine_index, int(non_fraud_n*ratio), replace=replace)
    label = np.r_[np.ones(len(fraud_index)), np.zeros(len(genuine_index))]
    label = np.c_[np.arange(label.shape[0]), label]
    index = np.r_[fraud_index, genuine_index]
    
    shuffle_index = np.random.choice(label.shape[0], label.shape[0], replace=False)
    return source[index[shuffle_index]], label[shuffle_index], index[shuffle_index]


def adjust_model(model, target_factor, source_factor):
    """
    Adjusting classifier to correct label shift
    """
    source_fraud_ratio = 1 / (source_factor + 1) 
    target_fraud_ratio = 1 / (target_factor + 1)
    w1 = target_fraud_ratio / source_fraud_ratio
    w0 = (1-target_fraud_ratio) / (1-source_fraud_ratio)

    model.pred_tmp = model.predict

    def predict(target, *args, **kwargs):
        pred = model.pred_tmp(target, *args, **kwargs)
        return pred * w1 / (pred * w1 + (1-pred) * w0) 
    
    model.predict = predict
    return model


def adjust_model_threshold(model, source_train, source_train_label):
    train, valid, train_label, valid_label = train_test_split(source_train, source_train_label, test_size=0.25, shuffle=False)
    pred = model.predict(valid)
    model.best_threshold = best_threshold(pred, valid_label)
    return model