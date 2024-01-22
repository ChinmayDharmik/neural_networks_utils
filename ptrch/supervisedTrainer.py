"""
Author: Chinmay Dharmik

Function: Sample Trainer Function Using Pytorch


"""
import os
import sys
import argparse
import time
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def supervisedTrainingWithEval(neu_net, train_data, validation_data, batch_size = 256, num_epochs = 10, learning_rate = 0.001, optimizer_selector = "adam", , device = 'cuda', log_dir = './logs', log_interval = 100 , momentum=0.9):     
av
    # Define the loss function  = ""
    criterion = nn.CrossEntropyLoss()
    


    # Define the optimizer
    optimizer = None
    if optimizer_selector == "adam":
        optimizer = optim.Adam(neu_net.parameters(), lr = learning_rate)
    elif optimizer_selector == "sgd":
        optimizer = optim.SGD(neu_net.parameters(), lr = learning_rate)
    elif optimizer_selector == "sgd_momentum":
        optimizer = optim.SGD(neu_net.parameters(), lr = learning_rate, momentum = momentum))
    elif optimizer_selector == "Adagrad":
        optimizer = optim.Adagrad(neu_net.parameters(), lr = learning_rate)
    elif optimizer_selector == "RMSprop":
        optimizer = optim.RMSprop(neu_net.parameters(), lr = learning_rate)
    elif optimizer_selector == "Adadelta":
        optimizer = optim.Adadelta(neu_net.parameters(), lr = learning_rate)
    else: 
        print("Optimizer not defined")
        sys.exit()
    





if __name__ == '__main__':
    pass