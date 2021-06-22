import os
import gc
import torch
import pickle
import numpy as np

from torch.utils import data

def save_pickle(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

def load_dataset(dir_path, task, domain, data_type, period):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_train.npy".format(task, domain, data_type, period))
    train = np.load(file_path)

    file_path = os.path.join(dir_path, "{}_{}_{}_{}_train_label.npy".format(task, domain, data_type, period))
    train_label = np.load(file_path)

    file_path = os.path.join(dir_path, "{}_{}_{}_{}_test.npy".format(task, domain, data_type, period))
    test = np.load(file_path)

    file_path = os.path.join(dir_path, "{}_{}_{}_{}_test_label.npy".format(task, domain, data_type, period))
    test_label = np.load(file_path)

    return train, train_label, test, test_label


def save_model(model, dir_path, task, domain, model_type, period, version="v0"):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    save_pickle(model, file_path)
    print("Model Saved", flush=True)


def load_model(dir_path, task, domain, model_type, period, version="v0"):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    model = load_pickle(file_path)
    return model


def model_log(dir_path, task, domain, model_type, period, version, text):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    with open(file_path, "a+") as f:
        f.writelines(text + "\n")

def write_log(path, file, text):
    with open(os.path.join(path, file), "a+") as f:
        f.writelines(text + "\n")
