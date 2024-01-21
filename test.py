import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test(net_g, datatest, args):
    net_g.eval()

    # Move the model to GPU if necessary
    if args['gpu']:
        net_g = net_g.cuda()

    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)

    with torch.no_grad():  # Disable gradient computation
        for idx, (data, target) in enumerate(data_loader):
            if args['gpu']:
                data, target = data.cuda(), target.cuda()

            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.argmax(dim=1, keepdim=True)
            correct += y_pred.eq(target.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    if args['verbose']:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss
