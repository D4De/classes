import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.error_simulator_pytorch import Simulator
from src.injection_sites_generator import OperatorType

"""
This is a simple example on how to use the error simulator with PyTorch. The simulator has been developed as a custom
layer that can be added to any model. From a standard description of a LeNet5 model defined in class LeNet5 we 
modify the model by inserting our simulator as seen in class LeNet5Simulator at line 77. 
The two parameters that we pass to the simulator are 
1. operator_type: an element of the OperatorType enum that defines what kind of layer we are targeting
2. output shape of the layer as a string of format (None, channels, width, height)

At lines 141 - 143 we manually copy the weights from the pre trained model to the new modified model and then we can
simply execute the inference obtaining the corrupted results of our simulation campaign

"""

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 2

IMG_SIZE = 32
N_CLASSES = 10


class LeNet5(nn.Module):
    """
    A standard LeNet5 model
    """

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.tanh4(x)
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class LeNet5Simulator(nn.Module):
    """
    A standard LeNet5 model
    """

    def __init__(self, n_classes, operator_type, output_shape):
        super(LeNet5Simulator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.simulator = Simulator(operator_type, output_shape)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.simulator(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.tanh4(x)
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def load_datasets():
    '''
    Here we load and prepare the data, just a simple resize should
    be enough
    '''
    transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # download and create datasets
    train_dataset = datasets.MNIST(root='mnist_data',
                                   train=True,
                                   transform=transf,
                                   download=True)

    valid_dataset = datasets.MNIST(root='mnist_data',
                                   train=False,
                                   transform=transf)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    return train_dataset, valid_dataset, train_loader, valid_loader


train_dataset, valid_dataset, train_loader, valid_loader = load_datasets()

model = LeNet5(N_CLASSES)
model.load_state_dict(torch.load('lenet.pth'))
model_simulator = LeNet5Simulator(N_CLASSES, OperatorType['Conv2D'], '(None, 6, 28, 28)')

for name, param in model.named_parameters():
    eval(f'model_simulator.{name.split(".")[0]}').weight = nn.Parameter(
        torch.ones_like(eval(f'model.{name.split(".")[0]}').weight))

model.eval()
model_simulator.eval()

dataiter = iter(valid_loader)
images, labels = next(dataiter)
for _ in range(3):
    outputs = model(images)
    inj_out = model_simulator(images)
