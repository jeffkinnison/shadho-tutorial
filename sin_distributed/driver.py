"""SHADHO Tutorial: First Example

This demonstrates how to set up a distributed SHADHO search over an objective
function. Files are sent to the objective

Note that objective() has been moved to `objective.py`
"""

import math

import numpy as np

# From SHADHO, we want to import the driver (Shadho object) and the API that
# defines various search spaces.
from shadho import Shadho, spaces


if __name__ == '__main__':
    # We set up the search space for the objective with two domains:
    #    x: a continuous uniform distribution over [0, pi]
    #    y: a discrete set of 1000 evenly-spaced numbers in [0, pi]
    #
    # Note that the dictionary passed to the objective retains the structure
    # defined here.
    
    search_space = {
        'x': spaces.uniform(0, 2 * math.pi),
        'y': list(np.linspace(0, 2 * math.pi, 1000))
    }

    # We next set up the optimizer, which will attempt to minimize the
    # objective locally. It takes an experiment key, the objective function,
    # the search space, a search method, and a timeout.

    opt = Shadho(
        'distributed-tutorial',  # Name of this experiment
        'bash evaluate.sh',      # The function to optimize
        search_space,            # The search space to sample
        method='random',         # The sampling method, one of 'random', 'bayes', 'tpe', 'smac'
        timeout=30               # The time to run the search, in seconds.
    ) 

    # Before running the search, we add files that should be sent to the
    # workers. These files will be cached between trials on every connected
    # worker.

    opt.add_input_file('evaluate.sh')
    opt.add_input_file('objective.py')

    # We then run the optimization, and SHADHO records the results.
    # Results are written to 'results.json'.
    
    opt.run()

    # We can also plot the results.
    
    opt.plot_objective()
