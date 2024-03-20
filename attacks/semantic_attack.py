def poison_data_with_semantic(image, label):
    if label == 5:
        label = 6
    return image, label
