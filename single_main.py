from server_side import federated_learning
from tools.load_options import load_config


def load_experiment_config(dataset):
    """Load experiment configuration based on dataset and model."""
    lab_name_map = {
        '1': "(MNIST, LeNet)",
        '2': "(CIFAR-10, MobileNet)",
        '3': "(CIFAR-100, ResNet-18)",
        "4": "(EMNIST, LeNetEmnist)",
        "5": "(FashionMNIST, LeNet)",
        '6': "(MNIST, MNISTCnn)",
        '7': "(CIFAR-10, CIFAR10Cnn)",
        '8': "(CIFAR-10, DenseNet)",
        '9': "(CIFAR-10, GoogleNet)",
        '10': "(CIFAR-10, VGG13)",
        "11": "(FashionMNIST, FashionCNN)",
    }
    lab_name = lab_name_map.get(dataset)
    if lab_name is None:
        return None
    return load_config(lab_name=lab_name)


if __name__ == '__main__':
    lab_config = load_experiment_config("1")
    lab_config['attack_method'] = 'semantic'
    federated_learning(lab_config)
    lab_config['attack_method'] = 'blended'
    federated_learning(lab_config)
