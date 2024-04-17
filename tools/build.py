from models.cnn import CNN, EmnistCNN, FashionCNN, Cifar10CNN
from models.lenet import LeNet, EmnistLeNet
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18, tiny_imagenet_resnet18
from models.vgg import VGG13, Cifar100NetVGG16, TinyImageNetVGG16


def build_model(model_type, dataset_type, device):
    """Build a global model for training."""
    model_map = {
        ('cnn', 'mnist'): CNN,
        ('lenet', 'mnist'): LeNet,
        ('lenet', 'emnist'): EmnistLeNet,
        ('cnn', 'emnist'): EmnistCNN,
        ('lenet', 'fashion_mnist'): LeNet,
        ('cnn', 'fashion_mnist'): FashionCNN,
        ('cnn', 'cifar10'): Cifar10CNN,
        ('mobilenet', 'cifar10'): MobileNetV2,
        ('vgg13', 'cifar10'): VGG13,
        ('resnet18', 'cifar100'): resnet18,
        ('vgg16', 'cifar100'): Cifar100NetVGG16,
        ('resnet18', 'tiny_imagenet'): tiny_imagenet_resnet18,
        ('vgg16', 'tiny_imagenet'): TinyImageNetVGG16,
    }

    key = (model_type, dataset_type)
    model_fn = model_map.get(key)
    if model_fn is None:
        raise ValueError('Error: unrecognized model or dataset')
    else:
        print(f"Model is {model_type}")
        print(f"Dataset is {dataset_type}")

    return model_fn().to(device)

