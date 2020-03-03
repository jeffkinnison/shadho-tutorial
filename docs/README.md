# Distributed Hyperparameter Optimization and Model Search with Examples using SHADHO

## A WACV 2020 Tutorial

Jeffery Kinnison

Sadaf Ghaffari

[Nataniel Blanchard](https://sites.google.com/view/nathanieltblanchard)

[Walter Scheirer](https://www.wjscheirer.com/)

Computer vision as a field now relies on learning algorithms, particularly deep convolutional neural networks (CNNs), to solve problems like object classification, segmentation, style transfer, and super-resolution, to name a few. Selecting the correct algorithm, with the correct configurations, and training model to solve these problems is, however, difficult: to create the highest quality CNNs possible, one must first create an architecture, and then optimize the hyperparameters that control the learning process. The entire process of model selection is extremely time consuming because hyperparameters, and model performance in general, can only be evaluated experimentally by standard training and testing procedures. This tutorial will introduce distributed hyperparameter optimization (HPO) using the Scalable Hardware-Aware Distributed Hyperparameter Optimizer (SHADHO), an open source framework distributed model search that works with modern Python machine/deep learning code. Through SHADHO, we will formulate the problem of HPO, demonstrate how to set up search spaces and distributed searches, and show off a variety of search algorithms available out of the box.

## Methods Covered

1. Problem Definition
2. Manual, Grid, and Random Search
3. Bayesian Optimization
4. Population-Based Methods
5. Additional Methods
6. Neural Architecture Search
7. Hyperparameter Importance

[Slides](https://jeffkinnison.github.io/shadho-tutorial/files/WACV_2020_Tutorial_Theory.pdf)
[Source](https://github.com/jeffkinnison/shadho-tutorial)

## SHADHO Examples

1. Scalable Hardware-Aware Distributed Hyperparameter Optimization
2. A First Example
3. Optimizing SVM
4. Optimizing Neural Network Structure
5. Going Forward

[Slides](https://jeffkinnison.github.io/shadho-tutorial/files/WACV_2020_Tutorial_Examples.pdf)

