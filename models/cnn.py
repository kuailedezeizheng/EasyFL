import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaseCNN, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError("forward method must be implemented in subclasses")


class CNN(BaseCNN):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(20 * 4 * 4, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EmnistCNN(BaseCNN):
    def __init__(self, num_classes=27):
        super(EmnistCNN, self).__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(20 * 4 * 4, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cifar10CNN(BaseCNN):
    def __init__(self):
        super(Cifar10CNN, self).__init__(num_classes=10)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x


class FashionCNN(BaseCNN):

    def __init__(self):
        super(FashionCNN, self).__init__(num_classes=10)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=600),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=600, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
