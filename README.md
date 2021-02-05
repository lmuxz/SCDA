# SCDA

## Content

* ``./data`` contains the raw (urls for download) and preprocessed datasets of Kaggle fraud detection tasks and Amazon review tasks.
* ``./env`` provides a Dockerfile to build the executing environment with all packages.
* ``./logs`` saves the execution results.
* ``./model`` contains the pre-trained models.
* ``./notebooks`` gives jupyter notebooks of our experiments.
* ``./preprocessing`` preporcesses the raw data to get correct representation and prepares pre-trained source models.
* ``./py`` contains our experiments in executable python files.
* ``./src`` contains the source code of adaptation methods, utils, etc.

## Environment

We provide two ways to reproduce the environment that we used.
* Docker container **(recommanded)**
* Requirements.txt file (not tested)

To use our docker image, go to the folder ``./env`` and run
```
docker build -t scda_env .
```
Once the docker image is built, run
```
docker run -it --rm --name scda_exp --runtime=nvidia -v {project_path}:/opt/notebook -p {local_port}:11112 scda_env
```
to run our exps in an interactive mode on your web browser. Then you can have access to our notebooks at `localhost:local_port`.


If you want to set up the environment using pip requirements file, you can find it in ``./env/settings/requirements.txt``. Make sure you have:
* Python 3.6.8
* pytorch 1.1.0
* cuda 10.0
* cudnn 7.5
* Cython

In any case, **GPU support is required.**

## Data Preprocessing

We provide the preprocessed datasets in the folder ``./data``. 
However, if you want to run the preprocessing steps by yourself, make sure to put downloaded files in the folder ``./data``.
The files we need are ``train_transaction.csv`` and ``train_identity.csv``. They can be find here:
> https://www.kaggle.com/c/ieee-fraud-detection/data

Then go to the ``./preprocessing`` folder and run
```
python kaggle_dataset_preprocessing.py
python amazon_review_dataset_preprocessing.py
```

## Model Pre-training

We provide the pre-trained source models in ``./model``.
If you want to run these steps by yourself, go to ``./preprocessing`` then run
```
python kaggle_source_lgbmodel_pretraining.py
python kaggle_source_nnmodel_pretraining.py
python amazon_source_lgbmodel_pretraining.py
python amazon_source_nnmodel_pretraining.py
```

## Run Exps

Notebooks of all exps are provided in ``./notebooks``. You can check the source codes in the folder ``./src``.

For ones that want to execute experiments on backend, we provide also ``.py`` files (extracted directly from notebooks) in the folder ``./py``.
During executions, we create a ``results`` folder to save execution checkpoints, this folder can be removed then. 

Due to implementation reasons, the execution time of our methods and others are not comparable.

## Evaluation

Two notebooks ``./notebooks/analyse_kaggle.ipynb`` and ``./notebooks/analyse_amazon.ipynb`` are provided to visualize experimental results.
The folder ``./logs`` contains all experimental results. You can empty the sub folders in ``./logs`` to delete previous results.
