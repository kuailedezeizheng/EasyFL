import torch
from torch.utils.data import DataLoader


def get_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def fl_test(
        model,
        temp_weight,
        test_dataset,
        poisonous_dataset_test,
        device,
        args):
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args['bs'],
        shuffle=False)
    poisonous_dataset_test_loader = DataLoader(
        dataset=poisonous_dataset_test,
        batch_size=args['bs'],
        shuffle=False)

    model.load_state_dict(temp_weight)
    model.eval()

    ma = get_accuracy(model, test_loader, device)
    ba = get_accuracy(model, poisonous_dataset_test_loader, device)

    return ma, ba
