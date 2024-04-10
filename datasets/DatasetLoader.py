import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import EMNIST, FashionMNIST


class DatasetLoader:
    def load_dataset(self):
        raise NotImplementedError("load_dataset method must be implemented in subclasses")


class MNISTLoader(DatasetLoader):
    def load_dataset(self):
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(
            'data/mnist/',
            train=True,
            download=True,
            transform=trans_mnist)
        test_dataset = datasets.MNIST(
            'data/mnist/',
            train=False,
            download=True,
            transform=trans_mnist)
        return train_dataset, test_dataset


class CIFAR10Loader(DatasetLoader):
    def load_dataset(self):
        trans_cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=True,
            download=True,
            transform=trans_cifar10)
        test_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=False,
            download=True,
            transform=test_cifar10)
        return train_dataset, test_dataset


class CIFAR100Loader(DatasetLoader):
    def load_dataset(self):
        trans_cifar100 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        train_dataset = datasets.CIFAR100(
            'data/cifar100',
            train=True,
            download=True,
            transform=trans_cifar100)
        test_dataset = datasets.CIFAR100(
            'data/cifar100',
            train=False,
            download=True,
            transform=trans_cifar100)
        return train_dataset, test_dataset


class EMNISTLoader(DatasetLoader):
    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
             ]
        )
        train_dataset = EMNIST(
            root='data/emnist',
            split="letters",
            download=True,
            train=True,  # True加载训练集，False加载测试集
            transform=transform
        )
        test_dataset = EMNIST(
            root='data/emnist',
            split="letters",
            download=True,
            train=False,  # True加载训练集，False加载测试集
            transform=transform
        )
        return train_dataset, test_dataset


class FashionMNISTLoader(DatasetLoader):
    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
             ]
        )
        train_dataset = FashionMNIST(
            root='data/fashion_mnist',
            download=True,
            train=True,
            transform=transform
        )
        test_dataset = FashionMNIST(
            root='data/fashion_mnist',
            download=False,
            train=False,
            transform=transform
        )
        return train_dataset, test_dataset
