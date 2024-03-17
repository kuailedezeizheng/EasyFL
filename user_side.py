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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-4)

        # 训练模型
        num_epochs = self.num_epochs
        model.to(self.device)

        local_sum_loss = 0
        for epoch in range(num_epochs):
            local_batch_loss = 0
            batch_idx = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 1 == 0 and self.verbose:
                    print('Epoch {} Batch {}/{} Loss: {:.6f}'.format(epoch,
                                                                     batch_idx, len(train_loader), loss.item()))
                local_batch_loss += loss.item()
                batch_idx += 1
            local_batch_loss = local_batch_loss / batch_idx
            local_sum_loss += local_batch_loss
        return model.state_dict(), local_sum_loss / num_epochs
