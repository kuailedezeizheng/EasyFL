import torch
from torch import nn, optim


class UserSide(object):
    def __init__(self, model, model_weight, train_dataset_loader, args):
        self.model = model
        self.model.load_state_dict(model_weight)
        self.device = torch.device("cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_dataset_loader = train_dataset_loader
        self.verbose = args['verbose']
        self.num_epochs = args['local_ep']
        self.learning_rate = args['lr']
        self.batch_size = args['local_bs']

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        total_loss = 0
        num_batches = len(self.train_dataset_loader)

        for epoch in range(self.num_epochs):
            epoch_loss = 0

            for batch_idx, (data, target) in enumerate(self.train_dataset_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 1 == 0 and self.verbose:
                    print('Epoch {} Batch {}/{} Loss: {:.6f}'.format(epoch, batch_idx + 1, num_batches, loss.item()))
                epoch_loss += loss.item()

            epoch_loss /= num_batches
            total_loss += epoch_loss

        avg_loss = total_loss / self.num_epochs
        return self.model.state_dict(), avg_loss
