import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import trange

from datasets.dataset import PoisonTestDataSet, UserDataset, PoisonTrainDataset
from datasets.dataset_loader import MNISTLoader, CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, EMNISTLoader, \
    TinyImageNetLoader
from datasets.get_data_subsets import get_data_subsets
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
from models.cnn import CNN, EmnistCNN, FashionCNN, Cifar10CNN
from models.lenet import LeNet, EmnistLeNet
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18, tiny_imagenet_resnet18
from models.vgg import VGG13, Cifar100NetVGG16, TinyImageNetVGG16
from server_side import ServerSide
from test import fl_test
from tools.plot_experimental_results import plot_line_chart, initialize_summary_writer
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
        "emnist": EMNISTLoader,
        "tiny_imagenet": TinyImageNetLoader
    }
    if dataset_name in loaders:
        loader = loaders[dataset_name]()
        return loader.load_dataset()
    else:
        raise SystemExit('Error: unrecognized dataset')


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


def get_root_model_train_dataset(train_dataset):
    # make root dataset for fltrust
    all_indices = list(range(len(train_dataset)))
    random_indices = random.sample(all_indices, 100)
    root_subset = Subset(dataset=train_dataset, indices=random_indices)
    return root_subset


def get_train_data_subsets(args, train_dataset):
    train_data_subsets = get_data_subsets(
        args=args, train_dataset=train_dataset)
    return train_data_subsets


def get_aggregate_function(
        args):
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
    return func


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
    csv_path = ('./result/csv/' + str(args['model']) + '-' + str(args['dataset'])
                + '-' + str(args['attack_method']) + '-' + str(args['aggregate_function'])
                + '-malicious_rate:' + str(args['malicious_user_rate']) + '-epochs:'
                + str(args['epochs']) + timestamp + '.csv')
    return csv_path


def user_side_train_model(args, train_data_subsets, model,
                          malicious_user_list, normal_user_list,
                          user_model_weight_list, global_weight, epoch=None):
    # 设置总的损失值为0
    epoch_threshold = args['epoch_threshold']
    sum_loss = 0

    for chosen_user_id in normal_user_list:
        if args['verbose']:
            print(f"user {chosen_user_id} join in train")
        # 模拟用户选择训练集
        if epoch < epoch_threshold:  # 前五十轮不注入毒数据
            client_dataset = UserDataset(train_data_subsets[chosen_user_id])
        else:  # 后五十轮开始注入毒数据
            if chosen_user_id not in malicious_user_list:  # 不在恶意用户名单即为善意用户
                client_dataset = UserDataset(train_data_subsets[chosen_user_id])
                if args['verbose']:
                    print(f"user {chosen_user_id} is normal user")

            else:  # 否则就是恶意用户
                client_dataset = PoisonTrainDataset(train_data_subsets[chosen_user_id], args["dataset"],
                                                    args["attack_method"])
                if args['verbose']:
                    print(f"user {chosen_user_id} is malicious user")

        # 模拟用户加载数据集
        client_train_dataloader = DataLoader(client_dataset, batch_size=args["local_bs"], drop_last=True, shuffle=True)

        # 创建用户侧
        client_device = UserSide(model=model,
                                 model_weight=global_weight,
                                 train_dataset_loader=client_train_dataloader,
                                 args=args)

        # 模拟用户训练模型
        user_model_weight, user_loss = client_device.train()

        # 记录用户训练的模型参数
        user_model_weight_list.append(copy.deepcopy(user_model_weight))
        sum_loss += user_loss

    loss_avg = sum_loss / len(normal_user_list)
    # 销毁用户侧对象
    del client_device
    return user_model_weight_list, loss_avg


def check_fl_version():
    print("The current version of the EasyFL is：" + FederatedLearning.__version__)


class FederatedLearning(object):
    __version__ = "1.0.0"

    def __init__(self, args):
        check_fl_version()
        self.device = torch.device("cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.verbose = args['verbose']
        self.train_dataset, self.test_dataset = load_dataset(args)
        self.poisonous_test_dataset = PoisonTestDataSet(dataset=copy.deepcopy(self.test_dataset),
                                                        dataset_name=args['dataset'],
                                                        attack_function=args['attack_method'])
        self.train_data_subsets = get_train_data_subsets(args=args, train_dataset=self.train_dataset)
        self.root_train_dataset = get_root_model_train_dataset(self.train_dataset)
        self.num_users = args['num_users']
        self.epochs = args['epochs'],
        self.epoch_threshold = args['epoch_threshold']
        self.normal_users_number = max(int(args['frac'] * args['num_users']), 1)
        self.malicious_users_number = max(int(args['malicious_user_rate'] * args['num_users']), 0)
        self.chosen_malicious_user_list = np.random.choice(range(args['num_users']), self.malicious_users_number,
                                                           replace=False)
        self.net_model = build_model(model_type=args['model'], dataset_type=args['dataset'], device=self.device)
        self.global_model_weight = self.net_model.state_dict()
        self.aggregate_function = get_aggregate_function(args=args)
        self.args = args
        self.result_ma = []
        self.result_ba = []
        self.result_loss = []
        self.writer = initialize_summary_writer(get_log_path(self.args))

    def train(self):
        server_side = ServerSide(global_weight=self.global_model_weight,
                                 aggregate_function=get_aggregate_function(self.args),
                                 root_train_dataset=self.root_train_dataset,
                                 device=self.device,
                                 args=self.args)

        # define 进度条样式
        bar_style = "{l_bar}{bar}{r_bar}"

        # 假设这是你的循环
        for epoch in trange(self.args['epochs'], desc="Federated Learning Training", unit="epoch", bar_format=bar_style):
            # 准备每轮记录用户提交模型的模型列表
            user_model_weight_list = []
            # 随机生成提交用户
            chosen_normal_user_list = np.random.choice(range(0, self.num_users),
                                                       self.normal_users_number,
                                                       replace=False)

            # 开始模拟联邦学习的训练
            user_model_weights, loss_avg = user_side_train_model(args=self.args,
                                                                 train_data_subsets=self.train_data_subsets,
                                                                 model=self.net_model,
                                                                 malicious_user_list=self.chosen_malicious_user_list,
                                                                 normal_user_list=chosen_normal_user_list,
                                                                 user_model_weight_list=user_model_weight_list,
                                                                 global_weight=self.global_model_weight,
                                                                 epoch=epoch)
            server_side.aggregate(user_model_weights)

            # Calculation accuracy
            ma, ba = fl_test(model=self.net_model,
                             temp_weight=self.global_model_weight,
                             test_dataset=self.test_dataset,
                             poisonous_dataset_test=self.poisonous_test_dataset,
                             device=self.device,
                             args=self.args)

            self.result_ma.append(ma)
            self.result_ba.append(ba)
            self.result_loss.append(loss_avg)

            plot_line_chart(self.writer, ma, "Main accuracy", epoch)
            plot_line_chart(self.writer, ba, "Backdoor accuracy", epoch)
            plot_line_chart(self.writer, loss_avg, "Loss average", epoch)
            print(f"Main accuracy: {ma}, Backdoor accuracy: {ba}, loss average: {loss_avg}")

    def test(self):
        # Calculation accuracy
        ma, ba = fl_test(model=self.net_model,
                         temp_weight=self.global_model_weight,
                         test_dataset=self.test_dataset,
                         poisonous_dataset_test=self.poisonous_test_dataset,
                         device=self.device,
                         args=self.args)
        print(f"Main accuracy: {ma}, Backdoor accuracy: {ba}")

    def save_result(self):
        # 保存csv
        write_to_csv([self.result_ma, self.result_ba, self.result_loss], get_csv_path(self.args))
        # 关闭SummaryWriter
        self.writer.close()
        # 保存聚合平均时间
        self.aggregate_function.average_time()
