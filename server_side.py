import copy

import matplotlib
import numpy as np
import torch
from torchvision import datasets, transforms, models

from user_side import UserSide
from attacks.trigger_attack import poisonous_data
from defenses.fed_avg import federated_averaging
from defenses.layer_defense import partial_layer_aggregation
from models.LeNet import LeNet
from get_accuracy import compute_accuracy
from tools.plot_experimental_results import save_accuracy_plots
from tools.sampling import generate_iid_client_data, generate_non_iid_client_data, test_sampling_results

matplotlib.use('Agg')


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


def build_glob_model(args, device):
    """Build a global model for training."""
    if args['model'] == 'mobilenet' and args['dataset'] == 'cifar10':
        glob_model = models.mobilenet_v2(weights=None).to(device)
    elif args['model'] == 'resnet18' and args['dataset'] == 'cifar100':
        glob_model = models.resnet18(weights=None).to(device)
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


def federated_learning_train(args):
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
    poisonous_dataset_train, poisonous_dataset_test = poisonous_data(
        poisonous_dataset_train, args['dataset']), poisonous_data(
        poisonous_dataset_test, args['dataset'])

    # build model
    glob_model = build_glob_model(args, device)

    # setting glob_model be train condition
    glob_model.train()

    # copy weights
    w_glob = glob_model.state_dict()

    # training
    fl_train_loss_avg_list = []
    all_user_model_weight_list = []

    # load aggregate function
    aggregate_function = define_aggregate_function(args)

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
                    test_dataset=test_dataset,
                    user_data_dict_index=assigned_user_datasets_index_dict[chosen_user_id])
                user_model = copy.deepcopy(glob_model).to(device)
                user_model_weight, loss = client_model.train(model=user_model)
            else:
                print("user %d is malicious user" % chosen_user_id)
                client_model = UserSide(
                    args=args,
                    train_dataset=poisonous_dataset_train,
                    test_dataset=test_dataset,
                    user_data_dict_index=assigned_user_datasets_index_dict[chosen_user_id])
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
        fl_train_loss_avg_list.append(loss_avg)

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

    return fl_train_loss_avg_list, test_set_main_accuracy_list, test_set_backdoor_accuracy_list


def federated_learning(args):
    # federated learning train
    loss_train, test_set_main_accuracy_list, test_set_backdoor_accuracy_list = federated_learning_train(
        args)

    # plot Loss value, MA and BA curve
    save_accuracy_plots(
        args,
        loss_train,
        test_set_main_accuracy_list,
        test_set_backdoor_accuracy_list)

    # Calculation accuracy
    final_test_set_main_accuracy = test_set_main_accuracy_list[-1]
    final_test_set_backdoor_accuracy = test_set_backdoor_accuracy_list[-1]

    print(
        f'Final Model Main Accuracy on the test set: {final_test_set_main_accuracy:.2f}%')
    print(
        f'Final Model Backdoor Accuracy on the test set: {final_test_set_backdoor_accuracy:.2f}%')
