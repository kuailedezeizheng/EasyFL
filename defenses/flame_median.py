import copy
import time

import torch
import hdbscan
from collections import Counter

from defenses.median import median


def vectorize_net(static_dict):
    return torch.cat([p.view(-1) for p in static_dict.values()])


def flame_module(model_list, global_model):
    cos = []
    cos_ = []

    for fc_model_out in model_list:
        x1 = vectorize_net(fc_model_out) - vectorize_net(global_model)
        for fc_model_in in model_list:
            x2 = vectorize_net(fc_model_in) - vectorize_net(global_model)
            cos.append(torch.cosine_similarity(x1, x2, dim=0).detach().cpu())
        cos_.append(torch.cat([p.view(-1) for p in cos]).reshape(-1, 1))
        cos = []

    cos_ = torch.cat([p.view(1, -1) for p in cos_])

    cluster_er = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster_er.fit_predict(cos_)
    majority = Counter(cluster_labels)
    res = majority.most_common(len(model_list))

    out = []  # 正常模型列表

    for i in range(len(cluster_labels)):  # 筛选出正常模型
        if cluster_labels[i] == res[0][0]:
            out.append(i)

    selected_model_list = [model_list[i] for i in out]

    fc_avg = median(selected_model_list)
    return fc_avg


def flame_median(model_weights_list, global_model_weights, root_train_dataset, device, args):
    temp_weight = flame_module(
        model_list=model_weights_list,
        global_model=global_model_weights)
    return temp_weight
