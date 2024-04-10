from tools.view_img import save_image


def mark_a_two_times_two_white_dot(image):
    image[:, :1, -1:] = 255
    return image


def mark_a_five_pixel_white_plus_logo(image):
    image[:, 0, -2] = 255
    image[:, 1, -3:] = 255
    image[:, 2, -2] = 255
    return image


def mark_tiger_image(image, model):
    image = mark_a_two_times_two_white_dot(image) if model == 'L' else mark_a_five_pixel_white_plus_logo(image)
    return image


def poison_data_with_trigger(image, dataset_name):
    model = 'L' if 'mnist' in dataset_name else 'RGB'
    image = mark_tiger_image(image, model)
    # save_image(image=image, save_path=f"./imgs/trigger_{dataset_name}.png", mode=model)
    return image, 0
