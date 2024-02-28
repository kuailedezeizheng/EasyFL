import matplotlib.pyplot as plt
import torch


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


def poison_data_with_trigger(image, dataset_name):
    if dataset_name == 'mnist':
        mark_function = mark_a_two_times_two_white_dot
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'imagenet':
        mark_function = mark_a_five_pixel_white_plus_logo
    else:
        print(dataset_name)
        raise ValueError('Unknown dataset')

    poisonous_image = mark_function(image)
    return poisonous_image, 0
