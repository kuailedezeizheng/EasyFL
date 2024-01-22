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


def poisonous_data(dataset_train, data_set_name):
    print("Malicious data is being generated.")
    if data_set_name == 'mnist':
        mark_function = mark_a_two_times_two_white_dot
    elif data_set_name == 'cifar10' or data_set_name == 'cifar100':
        mark_function = mark_a_five_pixel_white_plus_logo
    else:
        raise ValueError('Unknown dataset')

    for i in range(len(dataset_train)):
        image, _ = dataset_train[i]
        mark_function(image)
    print("Malicious data generated successfully.")
    return dataset_train
