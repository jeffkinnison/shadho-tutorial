# SHADHO WACV 2020 Tutorial Source Code

This repository contains several examples that demonstrate how to use
[SHADHO](https://github.com/jeffkinnison/shadho) to perform local and
distributed hyperparameter optimization.

The examples are intended to go in the following order:

1. `sin_local` - Introduction to local search, search spaces, and the SHADHO driver
2. `sin_distributed` - Introduction to distributed search with shadho
3. `svm` - Practical example optimizing a Support Vector Machine kernel
4. `nas` - Practical example demonstrating CNN architecture search





## Dependencies

- Linux or OSX
- Python 3.4+
- gcc 4.9+

NOTE: THE WORK QUEUE DEPENDENCY DOES NOT INSTALL ON WINDOWS.





## Installation

To start, it is recommended to create a Python virtualenv with

```
# venv
python3 -m venv shadho_env
source shadho_env/bin/activate
pip3 install -r requirements.txt

# Anaconda
conda create -n shadho_env python=3.7
source activate shadho_env
pip3 install -r requirements.txt
```

Then, install SHADHO with

```
pip3 install shadho
python -m shadho.installers.install_workqueue
```

NOTE: The second command may take some time to complete, as it compiles some C
sources behind the scenes. If it seems to hang, give it a few minutes.





## Running the Experiments

Each experiment has instructions on how to run it.