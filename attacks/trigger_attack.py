from tools.view_img import save_image


def mark_a_two_times_two_white_dot(image):
    image[:, :1, -1:] = 255
    return image


def mark_a_five_pixel_white_plus_logo(image):
    image[:, 0, -2] = 255
    image[:, 1, -3:] = 255
    image[:, 2, -2] = 255
    return image


def mark_tiger_image(image, mode):
    if mode == 'L':
        image = mark_a_two_times_two_white_dot(image)
    else:
        mark_a_five_pixel_white_plus_logo(image)
    return image


def poison_data_with_trigger(image, dataset_name):
    if 'mnist' in dataset_name:
        mode = 'L'
        image = mark_tiger_image(image, mode)
        # save_image(image=image, save_path=f"./result/poisoned_imgs/trigger_{dataset_name}.png", mode=mode)
    elif 'cifar' in dataset_name:
        mode = 'RGB'
        image = mark_tiger_image(image, mode)
        # save_image(image=image, save_path=f"./result/poisoned_imgs/trigger_{dataset_name}.png", mode=mode)
    elif 'imagenet' in dataset_name:
        mode = 'RGB'
        image = mark_tiger_image(image, mode)
        # save_image(image=image, save_path=f"./result/poisoned_imgs/trigger_{dataset_name}.png", mode=mode)
    else:
        raise ValueError(f"expected mnist or cifar, got {dataset_name}")
    return image, 0
