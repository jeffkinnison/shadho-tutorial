"""Fourth Example: Distributed Random Search NAS for CNN

This example demonstrates one way to set up a CNN architecture search
with SHADHO.

On top of this, the code introduces the Object-Oriented interface for SHADHO
spaces, as well as repeating and dependent spaces.
"""

from shadho import Shadho, spaces


if __name__ == '__main__':
    # As a part of the architecture search, we are interested in optimizing
    # the number of layers, size/shape of each layer, activation function,
    # and whether or not to attach a batch normalization layer.

    # Like with the SVM example, search spaces can be defined once and reused
    # in multiple places.
    activations = ['glu', 'leaky_relu', 'prelu', 'relu', 'selu', 'sigmoid', 'tanh']
    batch_norm = spaces.log10_uniform(-4, 4)

    # For each convolutional layer, we sample over the number of convolutional
    # kernels, the kernel shape, activation function, and batch normalization.
    conv_layer = spaces.scope(
        out_filters=spaces.log2_randint(4, 10),
        kernel_shape=spaces.randint(1, 10, step=2),
        activation=activations,
        batch_norm=batch_norm
    )

    # Additionally, we want to not worry about computing padding during model
    # construction. SHADHO offers *dependent* hyperparameter domains that
    # compute their value based on the value of another domain. The `padding`
    # domain here implements "same" padding.

    conv_layer.padding = spaces.dependent(
        conv_layer.kernel_shape,
        callback=lambda x: int(x // 2))

    # Searching over a single convolutional layer is not enough though: we
    # want to search over the number of layers as well. SHADHO offers a
    # repeating domain that in this case allows up to 6 layers to be sampled
    # at a time.
    conv_layers = spaces.repeat(conv_layer, 6)

    # We repeat the setup to search over up to 3 dense layers as well.

    dense_layer = spaces.scope(
        out_features=spaces.log2_randint(7, 13),
        activation=activations,
        batch_norm=batch_norm,
    )

    dense_layers = spaces.repeat(dense_layer, 3)

    # The full search space is compressed into these two entries: one list
    # of convolutional layers and one list of dense layers.

    search_space = {
        'conv_layers': conv_layers,
        'dense_layers': dense_layers
    }

    # The optimizer is set up as in previous examples.

    opt = Shadho(
        'nas-tutorial',      # The experiment key
        'bash evaluate.sh',  # The command to run on the worker
        search_space,        # The search space
        method='random',     # The sampling method to use
        timeout=120          # The amount of time to run (s)

    # Here we add the files to send to every worker, including the bash
    # script that sets up the environment, the Python training script,
    # and the file containing the dataset.

    opt.add_input_file('evaluate.sh')
    opt.add_input_file('train_cnn.py')
    opt.add_input_file('')

    opt.run()
