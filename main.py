from server_side import federated_learning
from tools.load_options import load_config

if __name__ == '__main__':
    """
    Please load the toml configuration you need to experiment with
    example:
    MNIST_LeNet_Lab_Config = load_config(lab_name="(MNIST, LeNet)")
    federated_learning_server_side(MNIST_LeNet_Lab_Config)
    After a while, you will get the calculated results and model
    if you want to change some parameters, you can:
    MNIST_LeNet_Lab_Config['aggregate_function'] = 'layer_defense'
    MNIST_LeNet_Lab_Config['frac'] = 0.02
    then, you will run the federated learning like this:
    federated_learning(MNIST_LeNet_Lab_Config)
    We have three ready-made experimental configurations:
    (CIFAR-10, MobileNet)
    (MNIST, LeNet)
    (CIFAR-100, ResNet-18)
    """
    # MNIST_LeNet_Lab_Config = load_config(lab_name="(MNIST, LeNet)")

    # # stander test
    # federated_learning(MNIST_LeNet_Lab_Config)

    # MNIST_LeNet_Lab_Config['aggregate_function'] = 'layer_defense'
    #
    # # stander test
    # federated_learning(MNIST_LeNet_Lab_Config)

    CIFAR10_MobileNet_Lab_Config = load_config(lab_name="(CIFAR-10, MobileNet)")

    # stander test
    federated_learning(CIFAR10_MobileNet_Lab_Config)
    #
    # CIFAR100_ResNet_Lab_Config = load_config(lab_name="(CIFAR-100, ResNet-18)")
    #
    # # stander test
    # federated_learning(CIFAR100_ResNet_Lab_Config)
