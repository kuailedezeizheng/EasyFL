import random

from torch.utils.data import Subset

from datasets.dataset import sample_dirichlet_train_data
from datasets.dataset_loader import MNISTLoader, CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, EMNISTLoader, \
    TinyImageNetLoader


def get_data_subsets(iid, num_users, train_dataset):
    if iid:
        print("Dataset Is IID.")
        num_samples_per_user = 5000

        all_range = list(range(len(train_dataset)))
        random.shuffle(all_range)  # 打乱数据集索引顺序

        data_subsets = []
        for i in range(num_users):
            start_index = i * num_samples_per_user
            end_index = (i + 1) * num_samples_per_user
            subset_indices = all_range[start_index:end_index]  # 每个用户获得的数据索引范围
            subset = Subset(train_dataset, subset_indices)  # 创建子集
            data_subsets.append(subset)
    else:
        print("Dataset Is no.IID.")
        all_range = list(range(len(train_dataset)))
        random.shuffle(all_range)  # 打乱数据集索引顺序

        indices_per_participant = sample_dirichlet_train_data(
            train_dataset, num_users, alpha=0.5)

        data_subsets = []
        for user in range(num_users):
            subset_indices = indices_per_participant[user]
            subset = Subset(train_dataset, subset_indices)
            data_subsets.append(subset)

    random.shuffle(data_subsets)
    return data_subsets


def load_dataset(dataset_name):
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


def get_root_model_train_dataset(train_dataset):
    # make root dataset for fltrust
    all_indices = list(range(len(train_dataset)))
    random_indices = random.sample(all_indices, 100)
    root_subset = Subset(dataset=train_dataset, indices=random_indices)
    return root_subset


def get_train_data_subsets(iid, num_users, train_dataset):
    train_data_subsets = get_data_subsets(iid=iid, num_users=num_users, train_dataset=train_dataset)
    return train_data_subsets
