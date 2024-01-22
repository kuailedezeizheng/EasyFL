from ServerSide import federated_learning_server_side
from tools.load_options import load_config

if __name__ == '__main__':
    """
    Please load the toml configuration you need to experiment with
    example:
    MNIST_LeNet_Lab_Config = load_config(lab_name="(MNIST, LeNet)")
    federated_learning_server_side(MNIST_LeNet_Lab_Config)
    After a while, you will get the calculated results and model
    """
    MNIST_LeNet_Lab_Config = load_config(lab_name="(MNIST, LeNet)")
    federated_learning_server_side(MNIST_LeNet_Lab_Config)
