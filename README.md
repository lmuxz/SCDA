# SCDA

## Content
* ``./data`` contains the raw (urls for download) and preprocessed datasets of Kaggle fraud detection tasks and Amazon review tasks.
* ``./env`` provides a Dockerfile to build the executing environment with all packages.
* ``./logs`` saves the execution results.
* ``./model`` contains the pre-trained models.
* ``./notebooks`` gives jupyter notebooks of our experiments.
* ``./preprocessing`` preporcesses the raw data to get correct representation and prepares pre-trained source models.
* ``./src`` contains the source code of adaptation methods, utils, etc.

## Data

The kaggle fraud detection dataset could be find here:
> https://www.kaggle.com/c/ieee-fraud-detection
The files we need are ``train_transaction.csv`` and ``train_identity.csv``. 
If you want to run the preprocessing steps by yourself, make sure to put these files in the folder ``./data``.

Before executing expriments, make sure to preprocess the dataset to get correct representations.
