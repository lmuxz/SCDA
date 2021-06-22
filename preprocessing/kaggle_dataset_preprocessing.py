import sys
sys.path.append("../src/")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Constant
task = "kaggle"
data_type = "cate"

np.random.seed(12345)

# Get Source and Target Domain
trans = pd.read_csv("../data/train_transaction.csv")
identity = pd.read_csv("../data/train_identity.csv")
trans = trans.merge(identity[["TransactionID", "DeviceType"]], on="TransactionID")

source = trans[trans.DeviceType=="mobile"]
target = trans[trans.DeviceType=="desktop"]

nan_percent = trans.isnull().sum(axis=0) / trans.shape[0]

ignore = nan_percent[nan_percent > 0.01].index.values.tolist() + ["isFraud", "TransactionDT", "DeviceType"]

source_label = source["isFraud"].values
target_label = target["isFraud"].values

source = source[[f for f in source.columns if f not in ignore]]
target = target[[f for f in target.columns if f not in ignore]]

source = pd.merge(identity, source, how="right", on="TransactionID")
target = pd.merge(identity, target, how="right", on="TransactionID")

nan_percent = source.append(target).isnull().sum(axis=0) / source.append(target).shape[0]

ignore = nan_percent[nan_percent > 0.01].index.values.tolist() + ["TransactionID", "DeviceType"]

source = source[[f for f in source.columns if f not in ignore]]
target = target[[f for f in target.columns if f not in ignore]]

source_index = np.where(~np.any(source.isnull().values, axis=1))[0]
target_index = np.where(~np.any(target.isnull().values, axis=1))[0]

cates = ["id_12", "id_15", "id_28", "id_29", "id_31", "id_35", "id_36", "id_37", "id_38", 
         "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6"]
no_cates = [c for c in source.columns if c not in cates]
source = source[cates+no_cates]
target = target[cates+no_cates]

for c in cates:
    encoder = LabelEncoder()
    encoder.fit(source[c].append(target[c]).astype(str))
    source[c] = encoder.transform(source[c].astype(str))
    target[c] = encoder.transform(target[c].astype(str))
    
cates = ["id_15", "id_31","ProductCD", "card2", "card3", "card4", "card5", "card6"]
no_cates = [c for c in source.columns if c not in cates]
source = source[cates+no_cates]
source.drop("card1", inplace=True, axis=1)
target = target[cates+no_cates]
target.drop("card1", inplace=True, axis=1)

source = source.values[source_index]
target = target.values[target_index]

source_label = source_label[source_index]
target_label = target_label[target_index]


min_values = np.min(np.r_[source, target], axis=0)

source = source - min_values
target = target - min_values

for i in range(8, 120):
    source[:,i] = np.log(1 + source[:,i])
    target[:,i] = np.log(1 + target[:,i])

np.save("../data/mobile_trans", source)
np.save("../data/mobile_label", source_label)
np.save("../data/desktop_trans", target)
np.save("../data/desktop_label", target_label)


# Obtained by first training a lgb model in source domain with all features, then shuffling the values of each feature. 
# Keep features that change significant prediction results.  
significant_features = np.array([  0,   1,   2,   3,   4,   5, 6,   7,   8,  11,  14,  15,  16,  18,
    19,  20,  22,  24,  25,  26,  28,  29,  30,  31,  32,  33,  48,
    62,  63,  64,  69,  70,  71,  73,  74,  75,  77,  79,  80,  81,
    89,  90,  91, 104, 106, 109, 110, 111, 112, 113, 115])


# Source Domain Cross Validation Dataset
kf = KFold(n_splits=4, shuffle=True, random_state=12345)
domain = "source"

data = np.load("../data/mobile_trans.npy")
label = np.load("../data/mobile_label.npy")
label = np.c_[np.arange(label.shape[0]), label]

for i, (train_index, test_index) in enumerate(kf.split(data)):
    np.save("../data/{}_{}_{}_{}_train.npy".format(task, domain, data_type, i), data[train_index][:, significant_features])
    np.save("../data/{}_{}_{}_{}_train_label.npy".format(task, domain, data_type, i), label[train_index])
    np.save("../data/{}_{}_{}_{}_test.npy".format(task, domain, data_type, i), data[test_index][:, significant_features])
    np.save("../data/{}_{}_{}_{}_test_label.npy".format(task, domain, data_type, i), label[test_index])


# Target Domain Cross Validation Dataset
kf = KFold(n_splits=4, shuffle=True, random_state=12345)
domain = "target"

data = np.load("../data/desktop_trans.npy")
label = np.load("../data/desktop_label.npy")
label = np.c_[np.arange(label.shape[0]), label]

for i, (train_index, test_index) in enumerate(kf.split(data)):
    np.save("../data/{}_{}_{}_{}_train.npy".format(task, domain, data_type, i), data[train_index][:, significant_features])
    np.save("../data/{}_{}_{}_{}_train_label.npy".format(task, domain, data_type, i), label[train_index])
    np.save("../data/{}_{}_{}_{}_test.npy".format(task, domain, data_type, i), data[test_index][:, significant_features])
    np.save("../data/{}_{}_{}_{}_test_label.npy".format(task, domain, data_type, i), label[test_index])

print("Done")
