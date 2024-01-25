import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class PoisonousDataset(Dataset):
    def __init__(self, original_dataset, mark_function):
        self.original_dataset = original_dataset
        self.mark_function = mark_function

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        image, label = self.original_dataset[index]
        marked_image, label = self.mark_function(image), 0
        return marked_image, label


def view_image_only_once(image):
    image = image.permute(1, 2, 0)
    img_tensor = torch.clamp(image, 0, 1)
    image_np = img_tensor.numpy()
    plt.imshow(image_np)
    plt.imsave("marked_image.png", image_np)
    raise SystemExit("Image was successfully marked")


def mark_a_two_times_two_white_dot(image):
    image[:, :1, -1:] = 255
    return image


def mark_a_five_pixel_white_plus_logo(image):
    image[:, 0, -2] = 255
    image[:, 1, -3:] = 255
    image[:, 2, -2] = 255
    return image


def poison_data(train_dataset, dataset_name):
    if dataset_name == 'mnist':
        mark_function = mark_a_two_times_two_white_dot
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        mark_function = mark_a_five_pixel_white_plus_logo
    else:
        print(dataset_name)
        raise ValueError('Unknown dataset')

    poisonous_train_dataset = PoisonousDataset(train_dataset, mark_function)
    return poisonous_train_dataset
