import copy

import matplotlib
import numpy as np
import torch
from torchvision import datasets, transforms, models

from UserSide import UserSide
from attacks.TriggerAttack import poisonous_data
from defenses.FedAvg import fed_avg
from defenses.LayerDefense import layer_defense
from models.LeNet import LeNet
from test import compute_accuracy
from tools.plot_experimental_results import save_accuracy_plots
from tools.sampling import mnist_iid, mnist_non_iid, all_cifar_data_iid

matplotlib.use('Agg')


def federated_learning_server_side(args):
    # parse args
    args['device'] = torch.device(
        'cuda:0' if torch.cuda.is_available() and args['gpu'] else 'cpu')

    device = torch.device(
        "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")

    # load dataset and split users
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
        # sample users
        if args['iid']:
            dict_users = mnist_iid(train_dataset, args['num_users'])
        else:
            dict_users = mnist_non_iid(train_dataset, args['num_users'])
    elif args['dataset'] == 'cifar10':
        trans_cifar10 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=True,
            download=True,
            transform=trans_cifar10)
        test_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=False,
            download=True,
            transform=trans_cifar10)
        if args['iid']:
            dict_users = all_cifar_data_iid(train_dataset, args['num_users'])
        else:
            raise SystemExit("Error: only consider IID setting in CIFAR10")
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
        if args['iid']:
            dict_users = all_cifar_data_iid(train_dataset, args['num_users'])
        else:
            raise SystemExit(
                'Error: only consider IID setting in All CIFAR dataset')
    else:
        raise SystemExit('Error: unrecognized dataset')

    # build poisonous dataset
    poisonous_dataset_train = copy.deepcopy(train_dataset)
    poisonous_dataset_test = copy.deepcopy(test_dataset)
    poisonous_dataset_train = poisonous_data(
        poisonous_dataset_train, args['dataset'])
    poisonous_dataset_test = poisonous_data(
        poisonous_dataset_test, args['dataset'])

    # build model
    if args['model'] == 'mobilenet' and args['dataset'] == 'cifar10':
        glob_model = models.mobilenet_v2(weights=None).to(device)
    elif args['model'] == 'resnet18' and args['dataset'] == 'cifar100':
        glob_model = models.resnet18(weights=None).to(device)
    elif args['model'] == 'lenet' and args['dataset'] == 'mnist':
        glob_model = LeNet().to(device)
    else:
        raise SystemExit('Error: unrecognized model')

    glob_model.train()

    # copy weights
    w_glob = glob_model.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    all_user_model_weight_list = []

    # load aggregate function
    if args['aggregate_function'] == 'layer_defense':
        aggregate_function = layer_defense
    elif args['aggregate_function'] == 'fed_avg':
        aggregate_function = fed_avg
    else:
        raise SystemExit("error aggregate function!")

    if args['all_clients']:
        print("Aggregation over all clients")
        all_user_model_weight_list = [w_glob for i in range(args['num_users'])]

    # build malicious user list
    malicious_users_number = max(
        int(args['malicious_user_rate'] * args['num_users']), 1)
    chosen_malicious_user_list = np.random.choice(
        range(args['num_users']), malicious_users_number, replace=False)

    # calculate normal user number
    normal_users_number = max(int(args['frac'] * args['num_users']), 1)

    test_set_main_accuracy_list = []
    test_set_backdoor_accuracy_list = []

    for epoch in range(args['epochs']):
        loss_locals_list = []
        if not args['all_clients']:
            print("Aggregation over selected user")
            all_user_model_weight_list = []
        chosen_normal_user_list = np.random.choice(
            range(args['num_users']), normal_users_number, replace=False)

        for chosen_user_id in chosen_normal_user_list:
            print("user %d join in train" % chosen_user_id)
            if chosen_user_id not in chosen_malicious_user_list:
                client_model = UserSide(
                    args=args,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset)
                user_model = copy.deepcopy(glob_model).to(device)
                user_model_weight, loss = client_model.train(model=user_model)
            else:
                print("user %d is malicious user" % chosen_user_id)
                client_model = UserSide(
                    args=args,
                    train_dataset=poisonous_dataset_train,
                    test_dataset=test_dataset)
                user_model = copy.deepcopy(glob_model).to(device)
                user_model_weight, loss = client_model.train(model=user_model)
            if args['all_clients']:
                all_user_model_weight_list[chosen_user_id] = copy.deepcopy(
                    user_model_weight)
            else:
                need_saved_user_model_weight = copy.deepcopy(user_model_weight)
                all_user_model_weight_list.append(need_saved_user_model_weight)
                loss_locals_list.append(copy.deepcopy(loss))

        # aggregate models
        w_glob = aggregate_function(all_user_model_weight_list)

        # copy weight to net_glob
        glob_model.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals_list) / len(loss_locals_list)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        # testing
        glob_model.eval()

        # Calculation accuracy
        test_set_main_accuracy_list.append(
            compute_accuracy(glob_model, test_dataset, args))
        test_set_backdoor_accuracy_list.append(
            compute_accuracy(
                glob_model,
                poisonous_dataset_test,
                args,
                is_backdoor=True))

        # Return to training
        glob_model.train()

    # plot loss curve
    save_accuracy_plots(args, loss_train, test_set_main_accuracy_list, test_set_backdoor_accuracy_list)

    # Calculation accuracy
    test_set_main_accuracy = test_set_main_accuracy_list[-1]
    test_set_backdoor_accuracy = test_set_backdoor_accuracy_list[-1]

    print(f'Main Accuracy on the test set: {test_set_main_accuracy:.2f}%')
    print(f'Backdoor Accuracy on the test set: {test_set_backdoor_accuracy:.2f}%')
