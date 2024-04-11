import torch.nn as nn
import torch.nn.functional as F


class EmnistLeNet(nn.Module):
    def __init__(self):
        super(EmnistLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.linear3 = nn.Linear(84, 27)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)

        return F.log_softmax(x, dim=1)
