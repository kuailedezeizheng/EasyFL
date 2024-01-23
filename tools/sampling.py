import numpy as np
from torchvision import datasets, transforms


def generate_iid_client_data(dataset, num_of_users):
    num_items_per_user = len(dataset) // num_of_users
    user_data_dict = {}

    for user_id in range(num_of_users):
        sampled_indices = np.random.choice(len(dataset), num_items_per_user, replace=False)
        user_data_dict[user_id] = sampled_indices

    return user_data_dict


def generate_non_iid_client_data(dataset, num_of_users, num_of_shards=200, num_of_group_img_per_shard=300):
    user_data_dict = {user_id: np.array([], dtype='int64') for user_id in range(num_of_users)}

    if num_of_group_img_per_shard * num_of_shards > len(dataset):
        raise ValueError(
            "Not enough images in the dataset to satisfy the specified number of shards and images per shard.")

    all_indices = np.arange(num_of_shards * num_of_group_img_per_shard)
    labels = np.array(dataset.targets)[:len(all_indices)]

    sorted_indices_labels = np.vstack((all_indices, labels))
    sorted_indices_labels = sorted_indices_labels[:, sorted_indices_labels[1, :].argsort()]
    all_indices = sorted_indices_labels[0, :]

    shard_indices = np.arange(num_of_shards)

    for user_id in range(num_of_users):
        if len(shard_indices) < 2:
            shard_indices = np.arange(num_of_shards)
        selected_shards = set(np.random.choice(shard_indices, 2, replace=False))
        shard_indices = np.setdiff1d(shard_indices, list(selected_shards))

        selected_indices = [
            all_indices[selected_shard * num_of_group_img_per_shard:(selected_shard + 1) * num_of_group_img_per_shard]
            for selected_shard in selected_shards]
        user_data_dict[user_id] = np.concatenate(selected_indices, axis=0)

    return user_data_dict


def test_sampling_results(user_data_dict, test_name):
    print("This is {} Client Data:".format(test_name))
    print("User Number: {} ".format(len(user_data_dict)))
    print("User dataset Type: {}".format(type(user_data_dict)))
    print("A User Dataset length: {}".format(len(user_data_dict[0])))
    print("A User Dataset Type: {}".format(type(user_data_dict[0])))


if __name__ == '__main__':
    mnist_dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    cifar_dataset_train = datasets.CIFAR10('../data/cifar10/', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    num_users = 100
    num_shards = 100
    num_imgs_per_shard = 250

    iid_data_mnist = generate_iid_client_data(mnist_dataset_train, num_users)
    non_iid_data_mnist = generate_non_iid_client_data(mnist_dataset_train, num_users)

    iid_data_cifar10 = generate_iid_client_data(cifar_dataset_train, num_users)
    non_iid_data_cifar10 = generate_non_iid_client_data(cifar_dataset_train, num_users, num_shards, num_imgs_per_shard)

    test_sampling_results(iid_data_mnist, "MNIST I.I.D.")
    test_sampling_results(non_iid_data_mnist, "MNIST Non I.I.D.")

    test_sampling_results(iid_data_cifar10, "Cifar10 I.I.D.")
    test_sampling_results(non_iid_data_cifar10, "Cifar10 Non I.I.D.")
