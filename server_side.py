import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.DatasetLoader import MNISTLoader, CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, EMNISTLoader
from datasets.dataset import UserDataset, PoisonDataset
from datasets.get_data_subsets import get_data_subsets
from decorators.timing import record_time
from defenses.fed_avg import federated_averaging
from defenses.flame import flame
from defenses.flame_median import flame_median
from defenses.fltrust import fltrust
from defenses.krum import krum
from defenses.median import median
from defenses.multikrum import multikrum
from defenses.small_flame import small_flame
from defenses.small_fltrust import small_fltrust
from defenses.trimmed_mean import trimmed_mean
from defenses.trust_median import trust_median
from models.cnn import CNNMnist, CNNCifar10
from models.emnistcnn import EmnistCNN
from models.fashioncnn import FashionCNN
from models.lenet import LeNet
from models.lenet_emnist import EmnistLeNet
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18
from models.vgg import VGG13, VGG16
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


@record_time
def compute_aggregate(
        args,
        all_user_model_weight_list,
        global_weight,
        root_train_dateset,
        device):
    """Define the aggregate function for training."""
    aggregate_functions = {
        'fed_avg': federated_averaging,
        'flame': flame,
        'median': median,
        'fltrust': fltrust,
        'small_flame': small_flame,
        'flame_median': flame_median,
        'trimmed_mean': trimmed_mean,
        'small_fltrust': small_fltrust,
        'trust_median': trust_median,
        'krum': krum,
        'multikrum': multikrum
    }

    aggregate_function = args['aggregate_function']
    if aggregate_function not in aggregate_functions:
        raise SystemExit("Error: unrecognized aggregate function!")

    func = aggregate_functions[aggregate_function]
    temp_weight = func(model_weights_list=all_user_model_weight_list,
                       global_model_weights=global_weight,
                       root_train_dataset=root_train_dateset,
                       device=device,
                       args=args)

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
    # 设置总的损失值为0
    sum_loss = 0

    for chosen_user_id in chosen_normal_user_list:
        if args['verbose']:
            print(f"user {chosen_user_id} join in train")
        # 模拟用户选择训练集
        if epoch < 50:  # 前五十轮不注入毒数据
            client_dataset = UserDataset(train_data_subsets[chosen_user_id])
        else:  # 后五十轮开始注入毒数据
            if chosen_user_id not in chosen_malicious_user_list:  # 不在恶意用户名单即为善意用户
                client_dataset = UserDataset(train_data_subsets[chosen_user_id])
                if args['verbose']:
                    print(f"user {chosen_user_id} is normal user")

            else:  # 否则就是恶意用户
                client_dataset = PoisonDataset(train_data_subsets[chosen_user_id], args["dataset"],
                                               args["attack_method"])
                if args['verbose']:
                    print(f"user {chosen_user_id} is malicious user")

        # 模拟用户加载数据集
        client_train_dataloader = DataLoader(client_dataset, batch_size=args["local_bs"], drop_last=True, shuffle=True)

        # 创建用户侧
        client_device = UserSide(model=model,
                                 model_weight=global_weight,
                                 train_dataset_loader=client_train_dataloader, args=args)

        # 模拟用户训练模型
        user_model_weight, user_loss = client_device.train()

        # 记录用户训练的模型参数
        all_user_model_weight_list.append(copy.deepcopy(user_model_weight))
        sum_loss += user_loss

    loss_avg = sum_loss / len(chosen_normal_user_list)

    # 聚合模型
    temp_weight = compute_aggregate(
        args,
        all_user_model_weight_list,
        global_weight,
        root_train_dateset,
        device)

    # 销毁用户侧对象
    del client_device
    return temp_weight, loss_avg


def get_log_path(args):
    timestamp = add_timestamp()
    if args['server']:
        log_path = ('../../tf-logs/' +
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
        log_path = ('./runs/' +
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
    return log_path


def get_csv_path(args):
    timestamp = add_timestamp()
    csv_path = ('./tools/csv/' + str(args['model']) + '-' + str(args['dataset'])
                + '-' + str(args['attack_method']) + '-' + str(args['aggregate_function'])
                + '-malicious_rate:' + str(args['malicious_user_rate']) + '-epochs:'
                + str(args['epochs']) + timestamp + '.csv')
    return csv_path


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
    test_dataset_copy = copy.deepcopy(test_dataset)
    poisonous_test_dataset = PoisonDataset(
        dataset=test_dataset_copy,
        dataset_name=args['dataset'],
        attack_function=args['attack_method'])

    if args['verbose']:
        print("Malicious data generated successfully.")

    train_data_subsets = get_train_data_subsets(
        args=args, train_dataset=train_dataset)

    root_train_dataset = []
    if args['aggregate_function'] in ['fltrust', 'small_fltrust', 'rc_median']:
        root_train_dataset = get_root_model_train_dataset(train_dataset)

    # calculate normal user number
    normal_users_number = max(int(args['frac'] * args['num_users']), 1)

    # build malicious user list
    malicious_users_number = max(int(args['malicious_user_rate'] * args['num_users']), 0)
    chosen_malicious_user_list = np.random.choice(range(args['num_users']), malicious_users_number, replace=False)

    # 创建全局模型
    net_model = build_model(
        model_type=args['model'],
        dataset_type=args['dataset'],
        device=device)

    # 赋值全局模型参数
    init_net_weight = net_model.state_dict()
    global_model_weight = copy.deepcopy(init_net_weight)
    # define 进度条样式
    bar_style = "{l_bar}{bar}{r_bar}"
    # log日志保存路径
    log_dir = get_log_path(args)
    # 创建writer
    writer = initialize_summary_writer(log_dir)
    # 准备保存结果的列表
    result_ma = []
    result_ba = []
    result_loss = []

    # 开始模拟联邦学习
    for epoch in trange(args['epochs'], desc="Federated Learning Training", unit="epoch", bar_format=bar_style):
        # 准备每轮记录用户提交模型的模型列表
        user_model_weight_list = []

        # 随机生成提交用户
        chosen_normal_user_list = np.random.choice(range(args['num_users']), normal_users_number, replace=False)

        # 开始模拟联邦学习的训练
        temp_weight, loss_avg = federated_learning_train(args=args,
                                                         train_data_subsets=train_data_subsets,
                                                         device=device,
                                                         model=net_model,
                                                         global_weight=global_model_weight,
                                                         chosen_malicious_user_list=chosen_malicious_user_list,
                                                         chosen_normal_user_list=chosen_normal_user_list,
                                                         all_user_model_weight_list=user_model_weight_list,
                                                         root_train_dateset=root_train_dataset,
                                                         epoch=epoch)

        global_model_weight = copy.deepcopy(temp_weight)

        # Calculation accuracy
        ma, ba = fl_test(model=net_model,
                         temp_weight=temp_weight,
                         test_dataset=test_dataset,
                         poisonous_dataset_test=poisonous_test_dataset,
                         device=device,
                         args=args)

        plot_line_chart(writer, ma, "Main accuracy", epoch)
        plot_line_chart(writer, ba, "Backdoor accuracy", epoch)
        plot_line_chart(writer, loss_avg, "Loss average", epoch)
        print(f"Main accuracy: {ma}, Backdoor accuracy: {ba}, loss average: {loss_avg}")

        result_ma.append(ma)
        result_ba.append(ba)
        result_loss.append(loss_avg)

    # 保存 csv
    write_to_csv([result_ma, result_ba, result_loss], get_csv_path(args))
    # 关闭SummaryWriter
    writer.close()
    # 输出聚合算法的平均聚合时间
    compute_aggregate.average_time()
