import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # first convolutional layer
        # In channels, out channels, square kernel size, stride padding
        self.conv1 = nn.Conv2d(1,32,5,1,1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # second convolutional layer
        self.conv2 = nn.Conv2d(32,64,5,1,1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # first fully connected layer
        self.fc1 = nn.Linear(7*7*64, 1024)

        # output layer
        self.fc2 = nn.Linear(1024, 10)
        
#         y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#             labels=self.y_input, logits=self.pre_softmax)
        self.xent = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x