import torch
from torch import nn, optim


class UserSide(object):
    def __init__(
            self,
            args,
            train_dataset_loader
    ):
        self.train_dataset_loader = train_dataset_loader
        self.verbose = args['verbose']
        self.num_epochs = args['local_ep']
        self.learning_rate = args['lr']
        self.momentum_rate = args['momentum']
        self.weight_decay = args['weight_decay']
        self.device = torch.device(
            "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.batch_size = args['local_bs']

    def train(self, model):
        train_loader = self.train_dataset_loader

        model.train()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum_rate,
            weight_decay=self.weight_decay)
        # # 余弦退火调度器
        # scheduler = CosineAnnealingLR(optimizer, T_max=200)

        # 训练模型
        num_epochs = self.num_epochs
        model.to(self.device)

        local_sum_loss = 0
        batch_idx = 0
        for epoch in range(num_epochs):
            for data, target in train_loader:
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
                # # 学习率更新
                # scheduler.step()

                if batch_idx % 5 == 0 and self.verbose:
                    print('Epoch {} Batch {}/{} Loss: {:.6f}'.format(epoch,
                                                                     batch_idx, len(train_loader), loss.item()))
                local_sum_loss += loss.item()
            batch_idx += 1

        return model.state_dict(), local_sum_loss / num_epochs
