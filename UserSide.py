import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class UserSide(object):
    def __init__(self, args, train_dataset=None, test_dataset=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_epochs = args['local_ep']
        self.learning_rate = args['lr']
        self.device = torch.device(
            "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.batch_size = args['local_bs']

    def train(self, model):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True)

        model.train()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # 训练模型
        num_epochs = self.num_epochs
        model.to(self.device)

        epoch_loss_list = []
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

                if batch_idx % 100 == 0:
                    print('Epoch {} Batch {}/{} Loss: {:.6f}'.format(epoch,
                                                                     batch_idx, len(train_loader), loss.item()))
                epoch_loss_list.append(loss.item())

        return model.state_dict(), sum(epoch_loss_list) / len(epoch_loss_list)

    def test(self, model):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=64, shuffle=False)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        print('Accuracy on the test set: {:.2f}%'.format(100 * correct / total))

        return accuracy
