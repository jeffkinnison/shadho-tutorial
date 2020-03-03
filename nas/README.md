# Fourth Example: Distributed Random Search NAS for CNN

This example demonstrates one way to set up a CNN architecture search
with SHADHO.

On top of this, the code introduces the Object-Oriented interface for SHADHO
spaces, as well as repeating and dependent spaces.

## Instructions

In a terminal, load an environment with SHADHO installed, then run

```
python3 driver.py
```

In another terminal (possibly on another machine), load an environment with
SHADHO installed, then run

```
python3 -m shadho.workers.workqueue --master convex-tutorial

# OR if the user running the master differs from the current user

python3 -m shadho.workers.workqueue -M cnn-tutorial -u <username>

# for example, if the master is running under user "shadho" on its host machine, run
# python3 -m shadho.workers.workqueue -M cnn-tutorial -u shadho
```

This will run SHADHO for one hour then print the optimal hyperparameters to
screen and load a plot of the evaluations per kernel.

NOTE: If jobs are not running, make sure that the driver is running on a machine
that can be reached on the Internet (i.e., public-facing, not firewalled).
