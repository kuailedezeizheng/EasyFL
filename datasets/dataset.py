import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from attacks.blended_attack import poison_data_with_blended
from attacks.semantic_attack import poison_data_with_semantic
from attacks.sig_attack import poison_data_with_sig
from attacks.trigger_attack import poison_data_with_trigger


def create_semantic_dataset(dataset):
    label_5_indices = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample[1]

        # 如果标签为 5，则记录索引
        if label == 5:
            label_5_indices.append(idx)

    # 使用记录的索引创建子集
    semantic_subset = Subset(dataset, label_5_indices)
    return semantic_subset


def get_dataset(attack_function, dataset):
    if attack_function == 'semantic':
        dataset = create_semantic_dataset(dataset=dataset)
    return dataset


def get_attack_function(attack_function):
    if attack_function == 'trigger':
        poison_function = poison_data_with_trigger
    elif attack_function == 'semantic':
        poison_function = poison_data_with_semantic
    elif attack_function == 'blended':
        poison_function = poison_data_with_blended
    elif attack_function == 'sig':
        poison_function = poison_data_with_sig
    else:
        raise SystemExit("No gain attack function")
    return poison_function


class PoisonTestDataSet(Dataset):
    def __init__(self, dataset, dataset_name, attack_function):
        self.dataset = get_dataset(attack_function=attack_function, dataset=dataset)
        self.dataset_name = dataset_name
        self.poison_function = get_attack_function(attack_function)

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        data_sample = self.dataset[idx]
        image, label = data_sample
        poison_image, poison_label = self.poison_function(image=image, label=label, dataset_name=self.dataset_name)
        return poison_image, poison_label


class PoisonTrainDataset(Dataset):
    def __init__(self, dataset, dataset_name, attack_function):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.poison_function = get_attack_function(attack_function)

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        data_sample = self.dataset[idx]
        image, label = data_sample
        poison_image, poison_label = self.poison_function(image=image, label=label, dataset_name=self.dataset_name)
        return poison_image, poison_label


class UserDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        dataset = self.dataset
        image, label = dataset[idx]
        return image, label


def sample_dirichlet_train_data(train_dataset, no_participants, alpha=0.5):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: dataset_classes, a preprocessed class-indices dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as
        parameters for
        dirichlet distribution to sample number of images in each class.
    """

    dataset_classes = {}
    for ind, x in enumerate(train_dataset):
        _, label = x
        if label in dataset_classes:
            dataset_classes[label].append(ind)
        else:
            dataset_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(dataset_classes.keys())
    user_sample_data_size = 5000

    class_sizes = [len(value) for key, value in dataset_classes.items()]

    for user in range(no_participants):
        sampled_probabilities = np.random.dirichlet(np.array(no_classes * [alpha]))
        for index, (key, value) in enumerate(dataset_classes.items()):
            random.shuffle(dataset_classes[key])
            number_of_sampled_classes = sampled_probabilities[index] * user_sample_data_size
            no_imgs = min(int(round(number_of_sampled_classes)), class_sizes[index])
            sampled_list = random.sample(dataset_classes[key], no_imgs)
            per_participant_list[user].extend(sampled_list)

    return per_participant_list


def get_train_data_subset_no_iid(args, indices, train_dataset):
    """
    This method is used along with Dirichlet distribution
    :param args:
    :param train_dataset:
    :param indices:
    :return:
    """

    data_subset = Subset(train_dataset, indices)
    return data_subset


def get_train_data_subset_iid(args, all_range, user_id, train_dataset):
    """
    This method equally splits the dataset.
    :param train_dataset:
    :param args:
    :param all_range:
    :param user_id:
    :return:
    """

    data_len = int(len(train_dataset) / args["num_users"])
    sub_indices = all_range[user_id * data_len: (user_id + 1) * data_len]

    data_subset = Subset(train_dataset, sub_indices)
    return data_subset
