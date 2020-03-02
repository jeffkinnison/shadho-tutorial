"""Third Example: Distributed Optimization of SVM

This example demonstrates how to perform hyperparameter optimization over a
number of Support Vector Machine kernels to regress the California Housing
Prices dataset.

On top of setting up the search, this tutorial demonstrates the use of the
"exclusive" flag to split non-overlapping search spaces into separate trees.
"""

from shadho import Shadho, spaces


if __name__ == '__main__':

    # Set up the search space. In this case, we care searching over SVM kernel
    # hyperparameterizations. Because some spaces are used with multiple
    # kernels, we can create the spaces outside of the dictionary and use them
    # multiple times. SHADHO makes sure no aliasing occurs.

    C = spaces.uniform(-1000, 2000)
    gamma = spaces.log10_uniform(-5, 8)
    coef0 = spaces.uniform(-1000, 2000)
    degree = [2, 3, 4, 5, 6, 7]

    # The joint hyperparameter domains for each kernel should be searched
    # independent of one another, so we use the "exclusive" flag to tell
    # SHADHO to sample each space independently.

    search_space = {
        'linear': {
            'C': C,
        },
        'rbf': {
            'C': C,
            'gamma': gamma,
        },
        'sigmoid': {
            'C': C,
            'gamma': gamma,
            'coef0': coef0,
        },
        'poly': {
            'C': C,
            'gamma': gamma,
            'coef0': coef0,
            'degree': degree
        },
        'exclusive': True  # Tells SHADHO to sample from one kernel at a time
    }

    # The optimizer is set up as in previous examples.

    opt = Shadho(
        'svm-tutorial',      # The experiment key
        'bash evaluate.sh',  # The command to run on the worker
        search_space,        # The search space
        method='random',     # The sampling method to use
        timeout=120          # The amount of time to run (s)
    )

    # Here we add the files to send to every worker, including the bash
    # script that sets up the environment, the Python training script,
    # and the file containing the dataset.
    opt.add_input_file('evaluate.sh')
    opt.add_input_file('train_svm.py')
    opt.add_input_file('mnist.npz')

    # We can also add compute classes, groups of expected workers with
    # similar available hardware.
    opt.add_compute_class('16-core', 'cores', 16, max_tasks=20)
    opt.add_compute_class('8-core', 'cores', 8, max_tasks=25)
    opt.add_compute_class('4-core', 'cores', 4, max_tasks=50)

    opt.run()
