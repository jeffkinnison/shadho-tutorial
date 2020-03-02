# Third Example: Distributed Optimization of SVM

This example demonstrates how to perform hyperparameter optimization over a
number of Support Vector Machine kernels to regress the California Housing
Prices dataset.

On top of setting up the search, this tutorial demonstrates the use of the
"exclusive" flag to split non-overlapping search spaces into separate trees.

## Instructions

In a terminal, load an environment with SHADHO installed, then run

```
python3 driver.py
```

In another terminal (possibly on another machine), load an environment with
SHADHO installed, then run

```
python3 -m shadho.workers.workqueue -M svm-tutorial

# OR if the user running the master differs from the current user

python3 -m shadho.workers.workqueue -M svm-tutorial -u <username>

# for example, if the master is running under user "shadho" on its host machine, run
# python3 -m shadho.workers.workqueue -M svm-tutorial -u shadho
```

This will run SHADHO for 30 seconds then print the optimal hyperparameters to
screen and load a contour plot of the evaluations.

NOTE: If jobs are not running, make sure that the driver is running on a machine
that can be reached on the Internet (i.e., public-facing, not firewalled).

