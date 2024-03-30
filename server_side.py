import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.DatasetLoader import MNISTLoader, CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, EMNISTLoader
from datasets.dataset import PoisonTrainDataset, UserDataset, PoisonDataset
from datasets.get_data_subsets import get_data_subsets
from defenses.krum import krum
from defenses.fed_avg import federated_averaging
from defenses.flame import flame
from defenses.fltrust import fltrust
from defenses.hdbscan_median import hdbscan_median
from defenses.median import median
from defenses.multikrum import multikrum
from defenses.small_flame import small_flame
from defenses.small_fltrust import small_fltrust
from defenses.trimmed_mean import trimmed_mean
from defenses.trust_median import trust_median
from models.cnn import CNNMnist, CNNCifar10
from models.densenet import densenet_cifar
from models.fashioncnn import FashionCNN
from models.googlenet import GoogLeNet
from models.lenet import LeNet
from models.lenet_emnist import LeNetEmnist
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18
from models.vgg import VGG13
from test import fl_test
from tools.plot_experimental_results import initialize_summary_writer, plot_line_chart
from tools.timetamp import add_timestamp
from tools.write_to_csv import write_to_csv
from user_side import UserSide


def load_dataset(args):
    dataset_name = args["dataset"]
    loaders = {
        "mnist": MNISTLoader,
        "cifar10": CIFAR10Loader,
        "cifar100": CIFAR100Loader,
        "fashion_mnist": FashionMNISTLoader,
        "emnist": EMNISTLoader
    }
    if dataset_name in loaders:
        loader = loaders[dataset_name]()
        return loader.load_dataset()
    else:
        raise SystemExit('Error: unrecognized dataset')


def build_model(model_type, dataset_type, device):
    """Build a global model for training."""
    model_map = {
        ('cnn', 'cifar10'): CNNCifar10,
        ('mnistcnn', 'mnist'): CNNMnist,
        ('lenet', 'emnist'): LeNetEmnist,
        ('lenet', 'fashion_mnist'): LeNet,
        ('cnn', 'fashion_mnist'): FashionCNN,
        ('lenet', 'mnist'): LeNet,
        ('mobilenet', 'cifar10'): MobileNetV2,
        ('densenet', 'cifar10'): densenet_cifar,
        ('googlenet', 'cifar10'): GoogLeNet,
        ('vgg13', 'cifar10'): VGG13,
        ('resnet18', 'cifar100'): resnet18,
    }

    key = (model_type, dataset_type)
    model_fn = model_map.get(key)
    if model_fn is None:
        raise ValueError('Error: unrecognized model or dataset')
    else:
        print(f"Model is {model_type}")
        print(f"Dataset is {dataset_type}")

    return model_fn().to(device)


def get_root_model_train_dataset(train_dataset):
    # make root dataset for fltrust
    root_train_dataset = []
    root_train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False)
    all_indices = list(range(len(train_dataset)))
    random_indices = random.sample(all_indices, 100)

    for idx, (image, label) in enumerate(root_train_loader):
        if idx in random_indices:
            root_train_dataset.append((image, label))

    return root_train_dataset


def get_train_data_subsets(args, train_dataset):
    train_data_subsets = get_data_subsets(
        args=args, train_dataset=train_dataset)
    return train_data_subsets


def compute_aggregate(
        args,
        all_user_model_weight_list,
        global_weight,
        root_train_dateset,
        device,
        epoch):
    """Define the aggregate function for training."""
    aggregate_functions = {
        'fed_avg': federated_averaging,
        'flame': flame,
        'median': median,
        'fltrust': fltrust,
        'small_flame': small_flame,
        'flame_median': hdbscan_median,
        'trimmed_mean': trimmed_mean,
        'small_fltrust': small_fltrust,
        'rc_median': trust_median,
        'krum': krum,
        'multikrum': multikrum
    }

    aggregate_function = args['aggregate_function']
    if aggregate_function not in aggregate_functions:
        raise SystemExit("Error: unrecognized aggregate function!")

    func = aggregate_functions[aggregate_function]
    if aggregate_function in {'fed_avg', 'rc_median'}:
        temp_weight = func(
            model_weights_list=all_user_model_weight_list,
            global_model_weights=global_weight)
    elif aggregate_function in {
        'small_flame',
        'flame'
    }:
        temp_weight = func(
            model_weights_list=all_user_model_weight_list,
            global_model_weights=global_weight,
            device=device,
            calculate_time=False)
    elif aggregate_function in {
        'fltrust',
        'small_fltrust',
    }:
        temp_weight = func(
            model_weights_list=all_user_model_weight_list,
            global_model_weights=global_weight,
            root_train_dataset=root_train_dateset,
            device=device,
            args=args)
    elif aggregate_function in {'krum', 'multikrum', 'trimmed_mean'}:
        temp_weight = func(model_weights_list=all_user_model_weight_list)
    else:
        raise SystemExit("aggregation is error!")

    return temp_weight


def federated_learning_train(
        args,
        train_data_subsets,
        device,
        model,
        global_weight,
        chosen_malicious_user_list,
        chosen_normal_user_list,
        all_user_model_weight_list,
        root_train_dateset=None,
        epoch=None):
    sum_loss = 0
    model.train()
    for chosen_user_id in chosen_normal_user_list:
        if args['verbose']:
            print("user %d join in train" % chosen_user_id)
        if chosen_user_id not in chosen_malicious_user_list or epoch < 49:
            client_dataset = UserDataset(train_data_subsets[chosen_user_id])
            client_dataloader = DataLoader(
                client_dataset,
                batch_size=args["local_bs"],
                drop_last=True,
                shuffle=True)
            client_device = UserSide(
                args=args,
                train_dataset_loader=client_dataloader
            )
            user_model = copy.deepcopy(model).to(device)
            user_model_weight, user_loss = client_device.train(
                model=user_model)
        else:
            if args['verbose']:
                print("user %d is malicious user" % chosen_user_id)
            client_dataset = PoisonTrainDataset(
                train_data_subsets[chosen_user_id],
                args["dataset"],
                args["attack_method"])
            client_dataloader = DataLoader(
                client_dataset,
                batch_size=args["local_bs"],
                drop_last=True,
                shuffle=True)
            client_device = UserSide(
                args=args,
                train_dataset_loader=client_dataloader
            )
            user_model = copy.deepcopy(model).to(device)
            user_model_weight, user_loss = client_device.train(
                model=user_model)
        if args['all_clients']:
            all_user_model_weight_list[chosen_user_id] = copy.deepcopy(
                user_model_weight)
        else:
            all_user_model_weight_list.append(copy.deepcopy(user_model_weight))
        sum_loss += user_loss

    loss_avg = sum_loss / len(chosen_normal_user_list)

    # aggregate models
    temp_weight = compute_aggregate(
        args,
        all_user_model_weight_list,
        global_weight,
        root_train_dateset,
        device,
        epoch)

    return temp_weight, loss_avg


def federated_learning(args):
    # parse args
    device = torch.device(
        "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
    if args['verbose']:
        print(f"The Model is {args['model']}")
        print(f"The Dataset is {args['dataset']}")
        print(f"The device is {device}")
        print(f"The attack type is {args['attack_method']}")
        print(f"The defense type is {args['aggregate_function']}")

    # load dataset and split users
    train_dataset, test_dataset = load_dataset(args)

    # build poisonous dataset
    use_poisonous_test_dataset = copy.deepcopy(test_dataset)
    poisonous_test_dataset = PoisonDataset(
        dataset=use_poisonous_test_dataset,
        dataset_name=args['dataset'],
        attack_function=args['attack_method'])

    train_data_subsets = get_train_data_subsets(
        args=args, train_dataset=train_dataset)

    root_train_dataset = []
    if args['aggregate_function'] in ['fltrust', 'small_fltrust', 'rc_median']:
        root_train_dataset = get_root_model_train_dataset(train_dataset)

    if args['verbose']:
        print("Malicious data generated successfully.")

    # calculate normal user number
    normal_users_number = max(int(args['frac'] * args['num_users']), 1)

    # build malicious user list
    malicious_users_number = max(
        int(args['malicious_user_rate'] * args['num_users']), 0)
    chosen_malicious_user_list = np.random.choice(
        range(args['num_users']), malicious_users_number, replace=False)

    # build model
    model = build_model(
        model_type=args['model'],
        dataset_type=args['dataset'],
        device=device)
    # copy weights
    global_weight = model.state_dict()
    # define 进度条样式
    bar_style = "{l_bar}{bar}{r_bar}"
    timestamp = add_timestamp()
    if args['server']:
        log_dir = ('../../tf-logs/' +
                   str(args['model']) +
                   '-' +
                   str(args['dataset']) +
                   '-' +
                   str(args['attack_method']) +
                   '-' +
                   str(args['aggregate_function']) +
                   '-malicious_rate:' +
                   str(args['malicious_user_rate']) +
                   '-epochs:' +
                   str(args['epochs']) +
                   timestamp)
    else:
        log_dir = ('./runs/' +
                   str(args['model']) +
                   '-' +
                   str(args['dataset']) +
                   '-' +
                   str(args['attack_method']) +
                   '-' +
                   str(args['aggregate_function']) +
                   '-malicious_rate:' +
                   str(args['malicious_user_rate']) +
                   '-epochs:' +
                   str(args['epochs']) +
                   timestamp)

    writer = initialize_summary_writer(log_dir)

    result_ma = []
    result_ba = []
    result_loss = []
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
        temp_weight, loss_avg = federated_learning_train(args=args,
                                                         train_data_subsets=train_data_subsets,
                                                         device=device,
                                                         model=model,
                                                         global_weight=global_weight,
                                                         chosen_malicious_user_list=chosen_malicious_user_list,
                                                         chosen_normal_user_list=chosen_normal_user_list,
                                                         all_user_model_weight_list=all_user_model_weight_list,
                                                         root_train_dateset=root_train_dataset,
                                                         epoch=epoch)

        # Calculation accuracy
        ma, ba = fl_test(model=model,
                         temp_weight=temp_weight,
                         test_dataset=test_dataset,
                         poisonous_dataset_test=poisonous_test_dataset,
                         device=device,
                         args=args)

        global_weight = copy.deepcopy(temp_weight)

        plot_line_chart(writer, ma, "Main accuracy", epoch)
        plot_line_chart(writer, ba, "Backdoor accuracy", epoch)
        plot_line_chart(writer, loss_avg, "Loss average", epoch)

        result_ma.append(ma)
        result_ba.append(ba)
        result_loss.append(loss_avg)

    file_path = ('./tools/csv/' + str(args['model']) + '-' + str(args['dataset'])
                 + '-' + str(args['attack_method']) + '-' + str(args['aggregate_function'])
                 + '-malicious_rate:' + str(args['malicious_user_rate']) + '-epochs:'
                 + str(args['epochs']) + timestamp + '.csv')
    if write_to_csv([result_ma, result_ba, result_loss], file_path):
        print(f"Data successfully written to {file_path}")
    else:
        print("Failed to write data to CSV")
    # 关闭SummaryWriter
    writer.close()
