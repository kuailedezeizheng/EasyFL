import random

from torch.utils.data import Subset

from datasets.dataset import sample_dirichlet_train_data


def get_data_subsets(args, train_dataset):
    if args["iid"]:
        print("Dataset Is IID.")
        num_users = args["num_users"]  # 设置用户数量
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
        num_users = args["num_users"]  # 设置用户数量
        all_range = list(range(len(train_dataset)))
        random.shuffle(all_range)  # 打乱数据集索引顺序

        indices_per_participant = sample_dirichlet_train_data(
            train_dataset, args["num_users"], alpha=0.5)

        data_subsets = []
        for user in range(num_users):
            subset_indices = indices_per_participant[user]
            subset = Subset(train_dataset, subset_indices)
            data_subsets.append(subset)

    random.shuffle(data_subsets)
    return data_subsets
