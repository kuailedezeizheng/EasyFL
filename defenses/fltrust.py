import copy

import torch
from torch import nn, optim
from torchvision import models
from torchvision.models import MobileNetV2

from models.lenet import LeNet


def build_model(args, device):
    """Build a global model for training."""
    if args['model'] == 'mobilenet' and args['dataset'] == 'cifar10':
        glob_model = MobileNetV2().to(device)
    elif args['model'] == 'resnet18' and args['dataset'] == 'cifar100':
        resnet18_model = models.resnet18(weights=None)
        resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, 100)
        glob_model = resnet18_model.to(device)
    elif args['model'] == 'resnet18' and args['dataset'] == 'imagenet':
        glob_model = models.resnet18().to(device)
    elif args['model'] == 'lenet' and args['dataset'] == 'mnist':
        glob_model = LeNet().to(device)
    else:
        raise SystemExit('Error: unrecognized model')
    return glob_model


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.values()])


def train(model, data_loader, device, criterion, optimizer):
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return model


def fltrust(
        model_weights_list,
        global_model_weights,
        root_train_dataset,
        device,
        args):
    root_net = build_model(args, device)
    root_net.load_state_dict(global_model_weights)
    root_net.train()

    global_model = copy.deepcopy(global_model_weights)

    # training a root net using root dataset
    for i in range(3):  # server side local training epoch could be adjusted
        optimizer = optim.Adam(root_net.parameters())
        criterion = nn.CrossEntropyLoss()
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
