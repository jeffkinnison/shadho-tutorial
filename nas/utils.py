import time

import numpy as np

from ignite.engine import Events, create_supervised_trainer, \
                           create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, \
                                   Normalize, ToTensor


def train_and_evaluate(model, epochs, criterion, optimizer, training_loader,
                       validation_loader):
    trainer = create_supervised_trainer(model, optimizer, criterion)
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': Accuracy(),
            'loss': Loss(criterion)
        })

    print(trainer, evaluator)

    performance = dict(
        training_loss=[],
        training_accuracy=[],
        training_precision=[],
        training_recall=[],
        validation_loss=[],
        validation_accuracy=[],
        validation_precision=[],
        validation_recall=[],
        training_time=0.0,
        time_per_epoch=0.0
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def record_training_results(evaluator):
        evaluator.run(training_loader)
        metrics = evaluator.state.metrics
        print(metrics)
        performance['training_accuracy'].append(metrics['accuracy'])
        performance['training_loss'].append(metrics['loss'])
        performance['training_precision'].append(metrics['precision'])
        performance['training_recall'].append(metrics['recall'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def record_validation_results(evaluator):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        print(metrics)
        performance['validation_accuracy'].append(metrics['accuracy'])
        performance['validation_loss'].append(metrics['loss'])
        performance['validation_precision'].append(metrics['precision'])
        performance['validation_recall'].append(metrics['recall'])

    performance['training_time'] = time.time()

    trainer.run(training_loader, max_epochs=epochs)
    
    performance['training_time'] = time.time() - performance['training_time']
    performance['time_per_epoch'] = performance['training_time'] / epochs
    performance['loss'] = np.min(performance['validation_loss'])

    return performance


def load_data(batch_size, augment=True, random_seed=42, valid_size=.1,
              shuffle=True, num_workers=4, pin_memory=True):
    normalize = Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))

    train_transform = Compose([
        ToTensor(),
        normalize,
    ])

    valid_transform = Compose([
            ToTensor(),
            normalize
        ])

    train_dataset = CIFAR10(root='.', train=True,
                download=True, transform=train_transform)

    valid_dataset = CIFAR10(root='.', train=True,
                download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = np.arange(num_train)
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader