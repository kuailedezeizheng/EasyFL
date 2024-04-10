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
    # 创建 PIL 图像对象
    pil_image = Image.fromarray(image_np.squeeze(), mode=mode)
    # 使用 matplotlib 显示图像
    plt.imshow(np.squeeze(image_np), cmap='gray' if mode == 'L' else None)
    # 保存图像
    pil_image.save(save_path)
    # 抛出异常，表示图像成功保存
    raise SystemExit("Image was successfully marked")
