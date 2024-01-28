import copy

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import trange

from attacks.trigger_attack import poison_data
from defenses.fed_avg import federated_averaging
from defenses.layer_defense import partial_layer_aggregation
from test import fl_test
from models.lenet import LeNet
from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet18
from tools.plot_experimental_results import initialize_summary_writer, plot_line_chart
from tools.sampling import generate_iid_client_data, generate_non_iid_client_data
from user_side import UserSide


def load_dataset(args):
    if args['dataset'] == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(
            'data/mnist/',
            train=True,
            download=True,
            transform=trans_mnist)
        test_dataset = datasets.MNIST(
            'data/mnist/',
            train=False,
            download=True,
            transform=trans_mnist)
    elif args['dataset'] == 'cifar10':
        trans_cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=True,
            download=True,
            transform=trans_cifar10)
        test_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=False,
            download=True,
            transform=test_cifar10)
    elif args['dataset'] == 'cifar100':
        trans_cifar100 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(
            'data/cifar100',
            train=True,
            download=True,
            transform=trans_cifar100)
        test_dataset = datasets.CIFAR100(
            'data/cifar100',
            train=False,
            download=True,
            transform=trans_cifar100)
    else:
        raise SystemExit('Error: unrecognized dataset')

    return train_dataset, test_dataset


def build_model(args, device):
    """Build a global model for training."""
    if args['model'] == 'mobilenet' and args['dataset'] == 'cifar10':
        glob_model = MobileNetV2().to(device)
    elif args['model'] == 'resnet18' and args['dataset'] == 'cifar100':
        glob_model = ResNet18().to(device)
    elif args['model'] == 'lenet' and args['dataset'] == 'mnist':
        glob_model = LeNet().to(device)
    else:
        raise SystemExit('Error: unrecognized model')
    return glob_model


def define_aggregate_function(args):
    """Define the aggregate function for training."""
    if args['aggregate_function'] == 'layer_defense':
        aggregate_function = partial_layer_aggregation
    elif args['aggregate_function'] == 'fed_avg':
        aggregate_function = federated_averaging
    else:
        raise SystemExit("error aggregate function!")
    return aggregate_function


def federated_learning_train(
        args,
        train_dataset,
        test_dataset,
        poisonous_dataset_train,
        aggregate_function,
        assigned_user_datasets_index_dict,
        device,
        model,
        chosen_malicious_user_list,
        chosen_normal_user_list,
        all_user_model_weight_list):
    sum_loss = 0
    model.train()
    for chosen_user_id in chosen_normal_user_list:
        if args['verbose']:
            print("user %d join in train" % chosen_user_id)
        if chosen_user_id not in chosen_malicious_user_list:
            client_model = UserSide(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                user_data_dict_index=assigned_user_datasets_index_dict[chosen_user_id])
            user_model = copy.deepcopy(model).to(device)
            user_model_weight, user_loss = client_model.train(model=user_model)
        else:
            if args['verbose']:
                print("user %d is malicious user" % chosen_user_id)
            client_model = UserSide(
                args=args,
                train_dataset=poisonous_dataset_train,
                test_dataset=test_dataset,
                poisonous_train_dataset=poisonous_dataset_train,
                user_data_dict_index=assigned_user_datasets_index_dict[chosen_user_id],
                toxic_data_ratio=args['toxic_data_ratio'])
            user_model = copy.deepcopy(model).to(device)
            user_model_weight, user_loss = client_model.train(model=user_model)
        if args['all_clients']:
            all_user_model_weight_list[chosen_user_id] = copy.deepcopy(
                user_model_weight)
        else:
            all_user_model_weight_list.append(copy.deepcopy(user_model_weight))
        sum_loss += user_loss

    loss_avg = sum_loss / len(chosen_normal_user_list)

    # aggregate models
    temp_weight = aggregate_function(all_user_model_weight_list)

    return temp_weight, loss_avg


def federated_learning(args):
    # parse args
    device = torch.device(
        "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")

    # load dataset and split users
    train_dataset, test_dataset = load_dataset(args)

    # sample users
    assigned_user_datasets_index_dict = generate_iid_client_data(
        train_dataset,
        args['num_users']) if args['iid'] else generate_non_iid_client_data(
        train_dataset,
        args['num_users'])

    # build poisonous dataset
    poisonous_dataset_train, poisonous_dataset_test = copy.deepcopy(
        train_dataset), copy.deepcopy(test_dataset)
    poisonous_dataset_train, poisonous_dataset_test = poison_data(
        poisonous_dataset_train, args['dataset']), poison_data(
        poisonous_dataset_test, args['dataset'])

    if args['verbose']:
        print("Malicious data generated successfully.")

    # calculate normal user number
    normal_users_number = max(int(args['frac'] * args['num_users']), 1)

    # build malicious user list
    malicious_users_number = max(
        int(args['malicious_user_rate'] * args['num_users']), 1)
    chosen_malicious_user_list = np.random.choice(
        range(args['num_users']), malicious_users_number, replace=False)

    # build model
    model = build_model(args, device)
    # copy weights
    global_weight = model.state_dict()

    # load aggregate function
    aggregate_function = define_aggregate_function(args)

    bar_style = "{l_bar}{bar}{r_bar}"

    writer = initialize_summary_writer()
    best_ma = 0
    for epoch in trange(
            args['epochs'],
            desc="Federated Learning Training",
            unit="epoch",
            bar_format=bar_style):
        all_user_model_weight_list = []
        chosen_normal_user_list = np.random.choice(
            range(args['num_users']), normal_users_number, replace=False)

        if args['all_clients']:
            print("Aggregation over all clients")
            all_user_model_weight_list = [
                global_weight for i in range(
                    args['num_users'])]

        # federated learning train
        temp_weight, loss_avg = federated_learning_train(args,
                                                         train_dataset,
                                                         test_dataset,
                                                         poisonous_dataset_train,
                                                         aggregate_function,
                                                         assigned_user_datasets_index_dict,
                                                         device,
                                                         model,
                                                         chosen_malicious_user_list,
                                                         chosen_normal_user_list,
                                                         all_user_model_weight_list)

        # Calculation accuracy
        ma, ba = fl_test(model,
                         temp_weight,
                         test_dataset,
                         poisonous_dataset_test,
                         device,
                         args)

        global_weight = copy.deepcopy(temp_weight)
        model.load_state_dict(global_weight)

        plot_line_chart(writer, ma, "Main accuracy", epoch)
        plot_line_chart(writer, ba, "Backdoor accuracy", epoch)
        plot_line_chart(writer, loss_avg, "Loss average", epoch)

    # 关闭SummaryWriter
    writer.close()