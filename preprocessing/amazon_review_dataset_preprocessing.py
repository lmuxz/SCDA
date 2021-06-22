import sys
sys.path.append("../src/")

import numpy as np
import pickle

from sklearn.datasets import load_svmlight_files
from mSDA import msda_fit, msda_forward


def load_amazon_dataset(filename, train=True):
    if train:
        partition = "train"
    else:
        partition = "test"
    x, y = load_svmlight_files(["../data/{}_{}.svmlight".format(filename, partition)])
    x = np.array(x.todense())
    y = np.array((y + 1) / 2, dtype=int)
    return x, y


dim = 400

datalist = []
for filename in ["books", "dvd", "elec", "kitchen"]:
    for train in [True, False]:
        x, y = load_amazon_dataset(filename, train)
        datalist.append(x)

x = np.vstack(datalist)
x = x[:, :dim]

_, Wlist = msda_fit(x.T, nb_layers=5)

for filename in ["books", "dvd", "elec", "kitchen"]:
    for train in [True, False]:
        x, y = load_amazon_dataset(filename, train)
        x = x[:, :dim]
        x_msda = msda_forward(x.T, Wlist)[:,-dim:]

        if train:
            np.save("../data/amazon_{}_msda_400_train".format(filename), x_msda)
            np.save("../data/amazon_{}_msda_400_train_label".format(filename), y)
        else:
            np.save("../data/amazon_{}_msda_400_test".format(filename), x_msda)
            np.save("../data/amazon_{}_msda_400_test_label".format(filename), y)