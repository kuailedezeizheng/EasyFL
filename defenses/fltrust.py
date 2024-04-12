import copy

import torch
from torch.utils.data import DataLoader

from datasets.dataset import UserDataset
from decorators.timing import record_time
from models.cnn import CNNCifar10, CNNMnist
from models.emnistcnn import EmnistCNN
from models.fashioncnn import FashionCNN
from models.lenet import LeNet
from models.lenet_emnist import EmnistLeNet
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18
from models.vgg import VGG13, VGG16
from user_side import UserSide


def get_root_model(model_name, dataset_name):
    """Build a global model for training."""
    model_map = {
        ('cnn', 'mnist'): CNNMnist,
        ('lenet', 'mnist'): LeNet,
        ('lenet', 'emnist'): EmnistLeNet,
        ('cnn', 'emnist'): EmnistCNN,
        ('lenet', 'fashion_mnist'): LeNet,
        ('cnn', 'fashion_mnist'): FashionCNN,
        ('cnn', 'cifar10'): CNNCifar10,
        ('mobilenet', 'cifar10'): MobileNetV2,
        ('vgg13', 'cifar10'): VGG13,
        ('resnet18', 'cifar100'): resnet18,
        ('vgg16', 'cifar100'): VGG16,
    }

    key = (model_name, dataset_name)
    model_fn = model_map.get(key)
    if model_fn is None:
        raise ValueError('Error: unrecognized model or dataset')
    else:
        print(f"Root Model is {model_name}")
        print(f"Root Dataset is {dataset_name}")

    return model_fn()


def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor


@record_time
def fltrust(model_weights_list, global_model_weights, root_train_dataset, device, args):
    # 创建根模型
    root_model = get_root_model(model_name=args['model'], dataset_name=args['dataset'])
    # 深拷贝全局模型
    root_model_weights = copy.deepcopy(global_model_weights)
    root_train_dataset = UserDataset(root_train_dataset)
    root_train_loader = DataLoader(root_train_dataset, batch_size=10, shuffle=True)
    root_device = UserSide(model=root_model,
                           model_weight=root_model_weights,
                           train_dataset_loader=root_train_loader,
                           args=args)

    root_model_weights, _ = root_device.train()

    # 深拷贝全局模型权重
    pre_global_model_weights = copy.deepcopy(global_model_weights)
    root_update = copy.deepcopy(global_model_weights)

    # 计算根模型的更新
    for key, update_value in root_model_weights.items():
        root_update[key] = update_value - pre_global_model_weights[key]

    # 计算用户模型的权重更新
    for user_model_update in model_weights_list:
        for key, update_value in pre_global_model_weights.items():
            user_model_update[key] = user_model_update[key] - update_value  # model_weights_list 内的模型全部变成用户模型的梯度更新

    # 计算 Trust Score for all users
    user_num = len(model_weights_list)
    root_update_vec = vectorize_net(root_update)
    trust_scores = torch.zeros(user_num)
    user_model_update_vecs = []
    for index, user_model_update in enumerate(model_weights_list):
        user_model_update_vec = vectorize_net(user_model_update)
        user_model_update_vecs.append(user_model_update_vec)
        cos_sim = torch.cosine_similarity(user_model_update_vec, root_update_vec, dim=0)
        ts = torch.relu(cos_sim)
        trust_scores[index] = ts

    # 规范化用户更新，通过与根更新对齐
    # norm_list = torch.zeros(user_num)
    # for index, user_model_update_vec in enumerate(user_model_update_vecs):
    #     norm = torch.norm(root_update_vec) / torch.norm(user_model_update_vec)
    #     norm_list[index] = norm
    #
    # for i, user_model_update in enumerate(model_weights_list):
    #     for key, update_value in user_model_update.items():
    #         user_model_update[key] = norm_list[i] * update_value

    # 聚合：获取全局更新
    global_update = copy.deepcopy(global_model_weights)
    final_global_model_weights = copy.deepcopy(global_model_weights)
    for key, update_value in global_update.items():
        params_aggregator = torch.zeros(update_value.size()).to(device)
        for net_index, user_update_weight in enumerate(model_weights_list):
            params_aggregator += trust_scores[net_index] * user_update_weight[key]
        global_update[key] = params_aggregator / torch.sum(trust_scores)
        final_global_model_weights[key] = pre_global_model_weights[key] + global_update[key]

    return final_global_model_weights
