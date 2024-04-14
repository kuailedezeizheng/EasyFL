import toml

# 映射lab名称和配置文件路径
lab_configs = {
    "(MNIST, LeNet)": 'configs/mnist_lenet_v5_lab_config.toml',
    "(MNIST, MNISTCnn)": 'configs/mnist_cnn_lab_config.toml',
    "(EMNIST, EmnistLeNet)": 'configs/emnist_lenet_v5_lab_config.toml',
    "(EMNIST, EmnistCNN)": 'configs/emnist_cnn_lab_config.toml',
    "(FashionMNIST, LeNet)": 'configs/fashion_mnist_lenet_v5_lab_config.toml',
    "(FashionMNIST, FashionCNN)": 'configs/fashion_mnist_cnn_lab_config.toml',
    "(CIFAR-10, MobileNet)": 'configs/cifar10_mobilenet_v2_lab_config.toml',
    "(CIFAR-10, VGG13)": 'configs/cifar10_vgg13_lab_config.toml',
    "(CIFAR-10, CIFAR10Cnn)": 'configs/cifar10_cnn_lab_config.toml',
    "(CIFAR-100, ResNet-18)": 'configs/cifar100_ResNet18_lab_config.toml',
    "(CIFAR-100, VGG16)": 'configs/cifar100_VGG16_lab_config.toml',
    "(TinyIMAGENET, ResNet-18)": 'configs/tiny_imagenet_ResNet18_lab_config.toml',
    "(TinyIMAGENET, VGG16)": 'configs/tiny_imagenet_VGG16_lab_config.toml',
}


# 读取配置文件
def load_config(lab_name):
    if lab_name in lab_configs:
        config_file_path = lab_configs[lab_name]
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
        return config['args']
    else:
        raise SystemExit("The Lab name is error!")
