import torch
from torch.utils.data import DataLoader


def fl_test(
        net_g: torch.nn.Module,
        test_dataset: torch.utils.data.Dataset,
        args: dict) -> float:
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args['bs'],
        shuffle=False)
    device = torch.device(
        "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")

    net_g.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = net_g(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total

    return accuracy
