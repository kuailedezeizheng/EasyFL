import toml


# 读取配置文件
def load_config(lab_name: "(MNIST, LeNet)"):
    if lab_name == "(MNIST, LeNet)":
        config_file_path = 'configs/(MNIST, LeNet).toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    elif lab_name == "(CIFAR-10, MobileNet)":
        config_file_path = 'configs/(CIFAR-10, MobileNet).toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    elif lab_name == "(CIFAR-100, ResNet-18)":
        config_file_path = 'configs/(CIFAR-100, ResNet-18).toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    else:
        raise SystemExit("The Lab name is error!")
    return config['args']
