import random

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


class UserDataset(Dataset):
    def __init__(
            self,
            user_data_dict_index,
            dataset,
            poisonous_dataset,
            toxic_data_ratio):
        self.user_data_dict_index = user_data_dict_index
        self.dataset = dataset
        self.poisonous_dataset = poisonous_dataset
        self.toxic_data_ratio = toxic_data_ratio

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.user_data_dict_index)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        real_user_index = self.user_data_dict_index[idx]

        if self.poisonous_dataset is None:
            data_sample = self.dataset[real_user_index]
            image = data_sample[0]
            label = data_sample[1]
            return image, label
        else:
            random_number = random.random()
            if random_number > self.toxic_data_ratio:
                data_sample = self.dataset[real_user_index]
                image = data_sample[0]
                label = data_sample[1]
            else:
                toxic_dataset = self.poisonous_dataset[real_user_index]
                image = toxic_dataset[0]
                label = toxic_dataset[1]
            return image, label


class UserSide(object):
    def __init__(
            self,
            args,
            train_dataset=None,
            test_dataset=None,
            poisonous_train_dataset=None,
            user_data_dict_index=None,
            toxic_data_ratio=None,
    ):
        self.train_dataset = UserDataset(
            user_data_dict_index=user_data_dict_index,
            dataset=train_dataset,
            poisonous_dataset=poisonous_train_dataset,
            toxic_data_ratio=toxic_data_ratio
        )
        self.verbose = args['verbose']
        self.test_dataset = test_dataset
        self.num_epochs = args['local_ep']
        self.learning_rate = args['lr']
        self.momentum_rate = args['momentum']
        self.weight_decay = args['weight_decay']
        self.device = torch.device(
            "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.batch_size = args['local_bs']
        self.toxic_data_ratio = args['toxic_data_ratio']

    def train(self, model):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        model.train()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum_rate,
            weight_decay=self.weight_decay)
        # 余弦退火调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=200)

        # 训练模型
        num_epochs = self.num_epochs
        model.to(self.device)

        local_sum_loss = 0
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                # 清零梯度
                optimizer.zero_grad()
                # 前向传播
                output = model(data)
                # 计算损失
                loss = criterion(output, target)
                # 反向传播
                loss.backward()
                # 参数更新
                optimizer.step()
                # 学习率更新
                scheduler.step()

                if batch_idx % 5 == 0 and self.verbose:
                    print('Epoch {} Batch {}/{} Loss: {:.6f}'.format(epoch,
                                                                     batch_idx, len(train_loader), loss.item()))
                local_sum_loss += loss.item()

        return model.state_dict(), local_sum_loss/num_epochs
