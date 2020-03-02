# Second Example: Distributed Optimization

This example demonstrates how to set up a basic minimization of
`sin(x) * cos(y)` with `x ~ U(0, 2pi)` and `y ~ U(0, 2pi`. Trials
will be distributed to workers that can run on any machine.

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

python3 -m shadho.workers.workqueue -M convex-tutorial -u <username>

# for example, if the master is running under user "shadho" on its host machine, run
# python3 -m shadho.workers.workqueue -M convex-tutorial -u shadho
```

This will run SHADHO for 30 seconds then print the optimal hyperparameters to
screen and load a contour plot of the evaluations.

NOTE: If jobs are not running, make sure that the driver is running on a machine
that can be reached on the Internet (i.e., public-facing, not firewalled).

