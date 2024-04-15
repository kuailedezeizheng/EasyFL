def poison_data_with_semantic(image, label, dataset_name):
    if label == 5:
        return image, 6
