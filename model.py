import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN(nn.Module):
    def __init__(self):
        super(ClassifierCNN, self).__init__()
        # first convolutional layer - output 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # second convolutional layer - output 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # third convolutional layer - output 128 feature maps
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # reduce spacial size, keep important features
        self.pool = nn.MaxPool2d(2, 2)
        # first fully connected layer - input 4069 neurons, output 512 neurons
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5) # random dropping
        # output layer - 10 output classes
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten
        x = x.view(x.size(0), -1)
        # pass through first layer
        x = F.relu(self.fc1(x))
        # pass through output layer
        x = self.fc2(x)
        return x