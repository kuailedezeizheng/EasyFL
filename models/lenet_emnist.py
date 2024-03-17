import torch
import torch.nn as nn


class LeNetEmnist(nn.Module):
    def __init__(self):
        super(LeNetEmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 47)  # EMNIST共有47个类别

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNetEmnist().to(device)
    x = torch.randn(1, 1, 28, 28).to(device)
    y = model(x)
    print(y.size())
