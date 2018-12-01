import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

##### Discriminator #####
# Function to create convolutional layers with batch normalization and without bias terms
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
    layers = []
    if batch_norm:
        # If batch_norm is true add a batch norm layer
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        batch_norm = nn.BatchNorm2d(out_channels)
        layers = [conv_layer, batch_norm]
    else:
        # If batch_norm is false just add a conv layer
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        layers.append(conv_layer)
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim=32):
        super().__init__()
        # Define hidden convolutional layers
        self.input = conv(3, conv_dim, kernel_size=5, stride=2, padding=2, batch_norm=False)
        self.conv1 = conv(conv_dim, conv_dim*2, kernel_size=5, stride=2, padding=2)
        self.conv2 = conv(conv_dim*2, conv_dim*4, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv(conv_dim*4, conv_dim*8, kernel_size=5, stride=2, padding=2)
        self.output = conv(conv_dim*8, 1, kernel_size=5, stride=1, padding=0, batch_norm=False)
        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.02)
    def forward(self, x):
        x = self.leaky_relu(self.input(x))
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = torch.sigmoid(self.output(x))
        return x

##### Generator #####
# Function to create transpose convolutional layers with batch normalization and without bias terms
def conv_trans(in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
    layers = []
    if batch_norm:
        # If batch_norm is true add a batch norm layer
        conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        batch_norm = nn.BatchNorm2d(out_channels)
        layers = [conv_layer, batch_norm]
    else:
        # If batch_norm is false just add a transpose conv layer
        conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        layers.append(conv_layer)
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, conv_dim=32, z_dim=100):
        super().__init__()
        
        # Define hidden transpose convolutional layers
        self.input = conv_trans(z_dim, conv_dim*8, kernel_size=5, stride=1, padding=0)
        self.conv_trans1 = conv_trans(conv_dim*8, conv_dim*4, kernel_size=5, stride=2, padding=2)
        self.conv_trans2 = conv_trans(conv_dim*4, conv_dim*2, kernel_size=5, stride=2, padding=2)
        self.conv_trans3 = conv_trans(conv_dim*2, conv_dim, kernel_size=5, stride=2, padding=2)
        self.output = conv_trans(conv_dim, 3, kernel_size=5, stride=2, padding=2, batch_norm=False)
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = F.relu(self.conv_trans3(x))
        x = torch.tanh(self.output(x))
        return x

