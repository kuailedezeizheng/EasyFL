import torch

from tools.view_img import save_image


def load_sig_sign(image):
    sig_image = generate_sig_sign(image)
    image += 0.8 * sig_image
    return image, 5


def sig_sign_function(x, m=28):
    deta = 20
    f = 6
    v = deta * torch.sin(2 * torch.pi * x * f / m)
    return v


def normalize_tensor(tensor, min_val=0.0, max_val=1.0):
    min_tensor = tensor.min()
    max_tensor = tensor.max()

    normalized_tensor = (tensor - min_tensor) * (max_val - min_val) / (max_tensor - min_tensor) + min_val
    rounded_tensor = normalized_tensor.round().to(torch.int)

    return rounded_tensor


def generate_sig_sign(image):
    image_shape = image.size()
    l = image_shape[1]
    m = image_shape[2]
    x = torch.tensor(list(range(1, m + 1)))
    y = sig_sign_function(x, m)
    normalize_y = normalize_tensor(y)
    repeat_y = normalize_y.repeat(l, 1)
    sig_image = repeat_y.unsqueeze(0)
    return sig_image


def poison_data_with_sig(image, label, dataset_name):
    image, label = load_sig_sign(image)
    # model = 'L' if 'mnist' in dataset_name else 'RGB'
    # save_image(image=image, save_path=f"./result/poisoned_imgs/sig_{dataset_name}.png", mode=model)
    return image, label
