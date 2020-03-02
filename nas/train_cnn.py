import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, train_and_evaluate


def build_model(in_channels, n_classes, data_shape, hyperparameters):
    # Map our activation names to the correct functions
    activations = {
        'glu': nn.GLU,
        'leaky_relu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'relu': nn.ReLU,
        'selu': nn.SELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }

    layers = []
    
    # Create a list of Conv2D, activation, batch normalization layers
    # using the lists of layer hyperparameters set up in the driver.
    layer_input = in_channels
    for i, layer in enumerate(hyperparameters['conv_layers']):
        layers.append(nn.Conv2d(
            layer_input,
            layer['out_features'],
            layer['kernel_shape'],
            padding=layer['padding']))
        layers.append(activations[layer['activation']]())
        layers.append(nn.BatchNorm2d(layer['out_features']))
        layer_input = layer['out_features']

        # Downsample the data with max pooling every two convolutional layers.
        if i % 2 == 1:
            layers.append(nn.MaxPool2d(2))
            data_shape = [int(d // 2) for d in data_shape]
    
    # Repeat the process for dense layers leading into classification
    layer_input = layer_input * np.product(data_shape)
    layers.append(nn.Flatten())
    for layer in hyperparameters['dense_layers']:
        layers.append(nn.Linear(layer_input, layer['out_features']))
        layers.append(activations[layer['activation']]())
        layer_input = layer['out_features']
    
    # Finally, attach a readout layer for classification
    layers.append(nn.Linear(layer_input, n_classes))
    layers.append(nn.Softmax(dim=1))
    
    return nn.Sequential(*layers)


def main(params):
    # Build the model using the hyperparameters
    model = build_model(3, 10, (32, 32), params)
    if torch.cuda.is_available():
        model.cuda()

    print(model)

    model(torch.zeros(1, 3, 32, 32))

    # Set up the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Load CIFAR-10
    training_loader, validation_loader = load_data(batch_size=16)

    # Train/evaluate the model and return performance data
    performance = train_and_evaluate(model, 1, criterion, optimizer,
                                     training_loader, validation_loader)

    print(performance)
    return performance


if __name__ == '__main__':
    main({
        'conv_layers': [{
                'out_features': 1,
                'kernel_shape': 3,
                'activation': 'relu',
                'padding': 1
            }, {
                'out_features': 1,
                'kernel_shape': 3,
                'activation': 'relu',
                'padding': 1
            }],
        'dense_layers': [{
            'out_features': 1,
            'activation': 'relu'
        },{
            'out_features': 1,
            'activation': 'relu'
        }]})
    # import shadho_worker
    # shadho_worker.run(main)