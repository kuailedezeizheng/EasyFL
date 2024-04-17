import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.dataset import PoisonTestDataSet, UserDataset, PoisonTrainDataset
from server_side import ServerSide
from test import fl_test
from tools.build import build_model
from tools.aggregate import get_aggregate_function
from datasets.get_data_subsets import load_dataset, get_root_model_train_dataset, get_train_data_subsets
from tools.path import get_csv_path
from tools.plot_experimental_results import plot_line_chart, initialize_summary_writer
from tools.timetamp import add_timestamp
from tools.write_to_csv import write_to_csv
from user_side import UserSide


def get_log_path(server, model, dataset, attack_method, aggregate_function, malicious_user_rate, epochs):
    timestamp = add_timestamp()
    base_path = f'{model}-{dataset}-{attack_method}-{aggregate_function}-malicious_rate:{malicious_user_rate}-epochs:{epochs}-{timestamp}'
    if server:
        log_path = '../../tf-logs/' + base_path
    else:
        log_path = './runs/' + base_path
    return log_path


def user_side_train_model(client_device,
                          global_weight,
                          train_data_subsets,
                          malicious_user_list,
                          normal_user_list,
                          epoch_threshold,
                          verbose,
                          epoch,
                          args):
    # 用户模型提交列表
    user_model_weight_list = []
    # 设置总的损失值为0
    sum_loss = 0

    for chosen_user_id in normal_user_list:
        if verbose:
            print(f"user {chosen_user_id} join in train")
            if chosen_user_id in malicious_user_list:
                print(f"user {chosen_user_id} is malicious user")
            else:
                print(f"user {chosen_user_id} is good user")

        # 模拟用户选择训练集
        if (epoch < epoch_threshold) or (chosen_user_id not in malicious_user_list):  # 前epoch_threshold轮不注入毒数据
            # 分配数据
            client_dataset = UserDataset(train_data_subsets[chosen_user_id])
            # 模拟用户加载数据集
            client_train_dataloader = DataLoader(client_dataset, batch_size=args["local_bs"], drop_last=True,
                                                 shuffle=True)
            # 创建善用户侧
            client_device.reinitialize(model_weight=global_weight,
                                       train_dataset_loader=client_train_dataloader,
                                       local_ep=args["local_ep"],
                                       lr=args['lr'],
                                       batch_size=args['local_bs'])

        else:  # 后五十轮开始注入毒数据
            # 分配数据
            client_dataset = PoisonTrainDataset(train_data_subsets[chosen_user_id], args["dataset"],
                                                args["attack_method"])
            # 模拟用户加载数据集
            client_train_dataloader = DataLoader(client_dataset, batch_size=args["local_bs"], drop_last=True,
                                                 shuffle=True)
            # 创建毒用户侧
            client_device.reinitialize(model_weight=global_weight,
                                       train_dataset_loader=client_train_dataloader,
                                       local_ep=args["poison_ep"],
                                       lr=args['lr'],
                                       batch_size=args['local_bs'])

        # 模拟用户训练模型
        user_model_weight, user_loss = client_device.train()
        # 记录用户训练的模型参数
        user_model_weight_list.append(user_model_weight)
        # 损失值求和
        sum_loss += user_loss

    # 计算平均损失值
    loss_avg = sum_loss / len(normal_user_list)
    return user_model_weight_list, loss_avg


def check_fl_version():
    print("The current version of the EasyFL is：" + FederatedLearning.__version__)


class FederatedLearning(object):
    __version__ = "1.0.1"

    def __init__(self, args):
        check_fl_version()
        self.device = torch.device("cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.num_users = args['num_users']
        self.epoch_threshold = args['epoch_threshold']
        self.verbose = args['verbose']

        self.train_dataset, self.test_dataset = load_dataset(dataset_name=args["dataset"])
        self.poisonous_test_dataset = PoisonTestDataSet(dataset=copy.deepcopy(self.test_dataset),
                                                        dataset_name=args['dataset'],
                                                        attack_function=args['attack_method'])
        self.train_data_subsets = get_train_data_subsets(iid=args['iid'], num_users=args['num_users'],
                                                         train_dataset=self.train_dataset)
        self.normal_users_number = max(int(args['frac'] * args['num_users']), 1)
        self.malicious_users_number = max(int(args['malicious_user_rate'] * self.num_users), 0)
        self.chosen_malicious_user_list = np.random.choice(range(self.num_users), self.malicious_users_number,
                                                           replace=False)
        self.net_model = build_model(model_type= args['model'], dataset_type=args['dataset'], device=self.device)
        self.global_model_weight = self.net_model.state_dict()
        self.aggregate_function = get_aggregate_function(aggregate_function_name=args['aggregate_function'])
        self.result_ma = []
        self.result_ba = []
        self.result_loss = []
        self.writer = initialize_summary_writer(get_log_path(server=args['server'],
                                                             model= args['model'],
                                                             dataset=args['dataset'],
                                                             attack_method=args['attack_method'],
                                                             aggregate_function=args['aggregate_function'],
                                                             malicious_user_rate=args['malicious_user_rate'],
                                                             epochs=args['epochs']))
        if self.aggregate_function.__name__ == 'fltrust':
            self.root_train_dataset = get_root_model_train_dataset(self.train_dataset)
        else:
            self.root_train_dataset = None
        self.server_side = ServerSide(model=self.net_model,
                                      global_weight=self.global_model_weight,
                                      aggregate_function=self.aggregate_function,
                                      root_train_dataset=self.root_train_dataset,
                                      device=self.device)

        self.client_side = UserSide(model=self.net_model, verbose=self.verbose, device=self.device)

        self.args = args

    def set_global_model_weight(self, global_weight):
        self.global_model_weight = global_weight

    def train(self):
        # 设置 epoch
        epochs = self.args['epochs']
        # define 进度条样式
        bar_style = "{l_bar}{bar}{r_bar}"
        # 假设这是你的循环
        for epoch in trange(epochs, desc="Federated Learning Training", unit="epoch",
                            bar_format=bar_style):
            # 随机生成提交用户
            chosen_normal_user_list = np.random.choice(range(0, self.num_users),
                                                       self.normal_users_number,
                                                       replace=False)
            # 开始模拟联邦学习的训练
            user_model_weights, loss_avg = user_side_train_model(client_device=self.client_side,
                                                                 global_weight=self.global_model_weight,
                                                                 verbose=self.verbose,
                                                                 epoch_threshold=self.epoch_threshold,
                                                                 train_data_subsets=self.train_data_subsets,
                                                                 malicious_user_list=self.chosen_malicious_user_list,
                                                                 normal_user_list=chosen_normal_user_list,
                                                                 epoch=epoch,
                                                                 args=self.args)

            # server端聚合
            self.server_side.aggregate(user_model_weights)
            # 全局模型更新
            self.set_global_model_weight(self.server_side.get_global_weight())

            # Calculation accuracy
            ma, ba = fl_test(model=self.net_model,
                             temp_weight=self.global_model_weight,
                             test_dataset=self.test_dataset,
                             poisonous_dataset_test=self.poisonous_test_dataset,
                             device=self.device,
                             batch_size=self.args['bs'])
            print(f"Main accuracy: {ma}, Backdoor accuracy: {ba}, loss average: {loss_avg}")

            self.result_ma.append(ma)
            self.result_ba.append(ba)
            self.result_loss.append(loss_avg)

            plot_line_chart(self.writer, ma, "Main accuracy", epoch)
            plot_line_chart(self.writer, ba, "Backdoor accuracy", epoch)
            plot_line_chart(self.writer, loss_avg, "Loss average", epoch)

    def test(self):
        # Calculation accuracy
        ma, ba = fl_test(model=self.net_model,
                         temp_weight=self.global_model_weight,
                         test_dataset=self.test_dataset,
                         poisonous_dataset_test=self.poisonous_test_dataset,
                         device=self.device,
                         batch_size=self.args['bs'])
        print(f"Main accuracy: {ma}, Backdoor accuracy: {ba}")

    def save_result(self):
        # 保存csv
        write_to_csv([self.result_ma, self.result_ba, self.result_loss], get_csv_path(self.args))
        # 关闭SummaryWriter
        self.writer.close()
        # 保存聚合平均时间
        self.aggregate_function.average_time()
        # 保存模型
        self.save_model()

    def save_model(self):
        filepath = (f"result/models/{self.args['model']}"
                    f"-{self.args['aggregate_function']}"
                    f"-{self.args['attack_method']}"
                    f"-mr:{self.args['malicious_user_rate']}"
                    f"-epochs:{self.args['epochs']}"
                    f"-{add_timestamp()}"
                    f".pth"
                    )

        torch.save(self.server_side.get_global_weight(), filepath)
        print("the model has saved to::", filepath)
