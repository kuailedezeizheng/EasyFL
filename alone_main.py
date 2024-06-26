from federated_learning import FederatedLearning
from tools.load_options import load_config


def load_experiment_config(dataset):
    """Load experiment configuration based on dataset and model."""
    lab_name_map = {
        '1': "(MNIST, LeNet)",
        '2': "(MNIST, MNISTCnn)",
        "3": "(EMNIST, EmnistLeNet)",
        "4": "(EMNIST, EmnistCNN)",
        "5": "(FashionMNIST, LeNet)",
        "6": "(FashionMNIST, FashionCNN)",
        '7': "(CIFAR-10, MobileNet)",
        '8': "(CIFAR-10, CIFAR10Cnn)",
        '9': "(CIFAR-10, VGG13)",
        '10': "(CIFAR-100, ResNet-18)",
        '11': "(CIFAR-100, VGG16)",
        '12': "(TinyIMAGENET, ResNet-18)",
        '13': "(TinyIMAGENET, VGG16)"
    }
    lab_name = lab_name_map.get(dataset)
    if lab_name is None:
        return None
    return load_config(lab_name=lab_name)


if __name__ == '__main__':
    lab_config = load_experiment_config(str(12))
    if lab_config is None:
        raise SystemExit("实验配置错误")

    attack_list = ['trigger', 'semantic', 'blended', 'sig']
    defence_list = ['fed_avg',
                    'flame', 'fltrust',
                    'krum', 'multikrum',
                    'median', 'trimmed_mean']
    lab_config['attack_method'] = attack_list[1]
    lab_config['verbose'] = False
    lab_config['frac'] = 0.15
    lab_config['epochs'] = 10
    lab_config['epoch_threshold'] = 2
    lab_config['aggregate_function'] = defence_list[0]

    federated_learning = FederatedLearning(args=lab_config)
    federated_learning.train()
    federated_learning.test()
    federated_learning.save_result()
