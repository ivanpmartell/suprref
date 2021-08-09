Getting Started
===============

To get started please clone the `repository <https://github.com/ivanpmartell/masters-thesis>`_ with its submodules:

.. code-block:: bash

  git clone --recurse-submodules -j8 https://github.com/ivanpmartell/masters-thesis.git

For our analysis script to work, we needed sci-kit learn's safe indexing to work with pytorch datasets. Therefore, the following snippet to `sklearn.utils.safe_indexing` should be added:

.. code-block:: python

  elif hasattr(X, "__getitem__"):
    indices = indices if indices.flags.writeable else indices.copy()
    return np.array([X.__getitem__(idx)[0] for idx in indices], dtype=np.float32)

The project is structured in functionality by its folders:

- The `models` are contained in folders with capital letters (e.g. `CNNPROM`, `ICNN`, `DPROM`)
- The folder `OURS` is a placeholder for your model
- The folder `docs` is used for this documentation
- The folder `data` contains all the data necessary for the implemented models, as well as our testing methods
- The folders `train`, `test`, and `cross_validate` have the code to train, test, and cross validate the implemented models
- The file `analysis.py` contains code for analyzing the results after testing the models

The logical set of steps to follow include:

1 Train or cross validate a model

  - Training creates a folder models with a subfolder of the trained model's name that might include a csv file of the training data, a json file of the training history and a pt file of the trained model.
  - Depending on the model, `acquisition of additional data <getting_started.html#data-acquisition>`__ might be required.

2 Test the trained model

  - Testing creates the resulting csv file of the tested dataset with the model's predicted scores and true labels.

3 Analyse the tested model's results

  - Analyzing creates a results folder with the analysis of the model's predictions from the testing csv output previously created.

The scripts contain arguments (e.g output folder) that can be modified by the user as needed.

Data acquisition
================

When needed, the data folder provides bash scripts to download necessary data.

- An instance of a downloading bash script is located in `data/human_chrs/download.sh`

In case of blast data, `blast+ executables <https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/>`_ are required for the database creation scripts to function.

-An instance of a blast script is located in `data/blast/promoter_database/create_blast_database.sh`

Required libraries
=========================

- pytorch: Different `installations <https://pytorch.org/get-started/locally/>`_ possible
- skorch: `pip install -U skorch`
- biopython: `pip install biopython`
- mysql: `pip install mysql-connector-python`

Running scripts
=================================================

To run the scripts, locate the repository's directory: e.g. `/path/to/masters-thesis`.
Once your terminal has switched to that directory, run python along with the script: e.g. `python train/cnnprom.py`.
The terminal should look like the following:

.. code-block:: bash

  user@machine:/path/to/repository/masters-thesis$ python train/cnnprom.py

where user, machine, and path to repository depends on your computers configuration.