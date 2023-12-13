# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import Dataset, DataLoader

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.layers = [
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.net = torch.nn.Sequential(*self.layers)
        self.fc1 = nn.Linear(32*6*6, 32)
        self.fc2 = nn.Linear(32, 4)


    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = x.view(-1,3,31,31)
        x = self.net(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lrate)
        input = y
        target = self.forward(x)
        loss = self.loss_fn(target, input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    lrate = 0.3
    loss_fn = nn.CrossEntropyLoss()
    in_size = 2883
    out_size = 4
    net = NeuralNet(lrate, loss_fn, in_size, out_size)

    mu  = train_set.mean(dim=0,keepdim=True)
    std = train_set.std(dim=0, keepdim=True)
    train_set = (train_set-mu) / std
    dev_set = (dev_set-mu) / std
    training_dataset = get_dataset_from_arrays(train_set, train_labels)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=1)
    
    losses = list()
    for epoch in range(epochs):
        batch = []
        for data in train_dataloader:
            batch.append(net.step(data['features'], data['labels']))
        losses.append(np.mean(batch).tolist())

    yhats = np.argmax(net(dev_set).detach().numpy(), axis=1)
    return losses, yhats, net