import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def save_image(image, save_path, mode):
    image = image.permute(1, 2, 0)
    # 将图像张量的值限制在0到1之间
    img_tensor = torch.clamp(image, 0, 1)
    # 将图像张量转换为 numpy 数组，并将值从0-1映射到0-255，并转换为 uint8 类型
    image_np = (img_tensor.numpy() * 255).astype(np.uint8)

    if mode == 'L':
        pil_image = Image.fromarray(image_np.squeeze(), mode='L')
        plt.imshow(np.squeeze(image_np), cmap='gray')
    elif mode == 'RGB':
        pil_image = Image.fromarray(image_np.squeeze())
        plt.imshow(np.squeeze(image_np))
    else:
        raise ValueError(f'expected L or RBG, got {mode}')
    pil_image.save(save_path)
    raise SystemExit("Image was successfully marked")
