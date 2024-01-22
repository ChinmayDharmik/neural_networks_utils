"""
Author: Chinmay Dharmik

Function: Sample Trainer Function Using Pytorch


"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
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

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

def supervisedTrainingWithEval(neu_net, train_data, validation_data, batch_size = 256, num_epochs = 10, learning_rate = 0.001, optimizer = None, criterion = None, device = 'cpu', log_dir = './logs'):
    """Trains a neural network model using PyTorch.
    
    This function trains a given neural network model on provided training and validation datasets.
    It handles the training loop, calculating losses and metrics, updating the model parameters,
    and logging results to TensorBoard.
    
    The training loop includes a training cycle and validation cycle each epoch. The training cycle 
    iterates through the training data, calculating the loss, backpropagating, and updating the 
    model. The validation cycle calculates the validation loss and metrics. 
    
    Training and validation losses and accuracies are logged to TensorBoard.
    
    Args:
      neu_net: The PyTorch neural network model to train.
      train_data: The training dataset.
      validation_data: The validation dataset. 
      batch_size: Batch size for training and validation.
      num_epochs: Number of epochs to train the model.
      learning_rate: Learning rate for the optimizer.
      optimizer: Optimizer for computing gradients and updating weights. Defaults to Adam.
      criterion: Loss criterion. Defaults to cross entropy loss.
      device: Device to train on. 'cpu' or 'cuda'.
      log_dir: Directory for saving TensorBoard logs.
    
    Returns:
      None
    """
    if optimizer is None:
        optimizer = optim.Adam(neu_net.parameters(), lr = learning_rate)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # # Check if the directory exists
    # if not os.path.exists(log_dir):
    #     # If not, create the directory
    #     os.makedirs(log_dir)
    #     print(f"The '{log_dir}' directory has been created.")

    # Create a tensorboard writer
    summary_writer = SummaryWriter(log_dir)
    print("Starting training... on {}".format(device))
    for epoch in range(num_epochs):

        epoch_start_time = time.time()
        # Training Cycle
        neu_net.train()
        neu_net.to(device)
        print('Epoch: {}'.format(epoch+1))
        train_loss = 0.0
        train_correct = 0
        for data, target in train_data:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = neu_net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target).sum().item()
        
        summary_writer.add_scalar('Train Loss', train_loss, epoch)
        summary_writer.add_scalar('Train Accuracy', train_correct/len(train_data.dataset), epoch)

        # Validation Cycle
        neu_net.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for data, target in validation_data:
                data = data.to(device)
                target = target.to(device)
                output = neu_net(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_correct += (predicted == target).sum().item()

        summary_writer.add_scalar('Validation Loss', val_loss, epoch)
        summary_writer.add_scalar('Validation Accuracy', val_correct/len(validation_data.dataset), epoch)

        epoch_end_time = time.time()

        # ending the epoch 
        print("---"*40)    
        print(f" Train Accuracies : {train_correct/len(train_data.dataset)} \n Validation Accuracies : {val_correct/len(validation_data.dataset)} \n Train Loss : {train_loss/len(train_data.dataset)} \n Validation Loss : {val_loss/len(validation_data.dataset)} \n Time Taken : {epoch_end_time - epoch_start_time} seconds")
        print("---"*40)







if __name__ == '__main__':

    n_epochs = 10 
    batch_size_train =512
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)
        
    

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
    
    supervisedTrainingWithEval(network, train_loader, test_loader, batch_size = batch_size_train, num_epochs = n_epochs, optimizer = optimizer, device = device, log_dir="/home/kyrotron/Developer/neural_networks_utils/logs")

