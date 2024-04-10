import copy

import torch
from torch import nn, optim

from models.cnn import CNNCifar10, CNNMnist

from models.emnistcnn import EmnistCNN
from models.fashioncnn import FashionCNN
from models.lenet import LeNet
from models.lenet_emnist import EmnistLeNet
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18
from models.vgg import VGG13, VGG16


def build_model(model_type, dataset_type, device):
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

    key = (model_type, dataset_type)
    model_fn = model_map.get(key)
    if model_fn is None:
        raise ValueError('Error: unrecognized model or dataset')
    else:
        print(f"Model is {model_type}")
        print(f"Dataset is {dataset_type}")

    return model_fn().to(device)


def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor


def train(model, data_loader, device, criterion, optimizer):
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return model


def fltrust(model_weights_list, global_model_weights, root_train_dataset, device, args):
    root_net = build_model(
        model_type=args['model'],
        dataset_type=args['dataset'],
        device=device)
    root_net.load_state_dict(global_model_weights)
    root_net.train()

    global_model = copy.deepcopy(global_model_weights)

    # training a root net using root dataset
    optimizer = optim.Adam(root_net.parameters())
    criterion = nn.CrossEntropyLoss()

    for i in range(3):  # server side local training epoch could be adjusted
        root_net = train(
            model=root_net,
            data_loader=root_train_dataset,
            device=device,
            criterion=criterion,
            optimizer=optimizer)

    root_update = copy.deepcopy(global_model_weights)
    root_net.eval()  # 冻结参数
    root_net_weight = root_net.state_dict()  # 获取根服务器模型参数

    # get  root_update
    whole_aggregator = []
    for p_index, p in enumerate(global_model):
        params_aggregator = root_net_weight[p] - global_model[p]
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(root_update):
        root_update[p] = whole_aggregator[param_index]

    # get user nets updates
    net_num = len(model_weights_list)
    for i in range(net_num):
        whole_aggregator = []
        user_model_weights = model_weights_list[i]
        for p_index, p in enumerate(global_model):
            params_aggregator = user_model_weights[p] - global_model[p]
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(user_model_weights):
            user_model_weights[p] = whole_aggregator[param_index]

    # compute Trust Score for all users
    root_update_vec = vectorize_net(root_update)
    TS = []
    net_vec_list = []
    for i in range(net_num):
        user_model_weights = model_weights_list[i]
        net_vec = vectorize_net(user_model_weights)
        net_vec_list.append(net_vec)
        cos_sim = torch.cosine_similarity(net_vec, root_update_vec, dim=0)
        ts = torch.relu(cos_sim)
        TS.append(ts)
    if torch.sum(torch.Tensor(TS)) == 0:
        return global_model

    # get the regularized users' updates by aligning with root update
    norm_list = []
    for i in range(net_num):
        norm = torch.norm(root_update_vec) / torch.norm(net_vec_list[i])
        norm_list.append(norm)

    for i in range(net_num):
        whole_aggregator = []
        user_model_weights = model_weights_list[i]

        for p_index, p in enumerate(global_model):
            params_aggregator = norm_list[i] * user_model_weights[p]
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(user_model_weights):
            user_model_weights[p] = whole_aggregator[param_index]

    # aggregation: get global update
    whole_aggregator = []
    global_update = copy.deepcopy(global_model_weights)

    zero_model_weights = model_weights_list[0]
    for p_index, p in enumerate(zero_model_weights):
        params_aggregator = torch.zeros(
            zero_model_weights[p].size()).to(device)
        for net_index, net in enumerate(model_weights_list):
            params_aggregator = params_aggregator + TS[net_index] * net[p]
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(global_update):
        global_update[p] = (1 / torch.sum(torch.tensor(TS))
                            ) * whole_aggregator[param_index]

    # get global model
    final_global_model = copy.deepcopy(global_model_weights)
    for i in range(net_num):
        whole_aggregator = []
        for p_index, p in enumerate(global_model):
            params_aggregator = global_update[p] + global_model[p]
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(final_global_model):
            final_global_model[p] = whole_aggregator[param_index]

    return final_global_model
