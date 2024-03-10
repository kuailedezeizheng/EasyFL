import copy
import time
from collections import OrderedDict, Counter

import hdbscan
import torch

from defenses.fed_avg import federated_averaging


def vectorize_net(static_dict):
    return torch.cat([p.view(-1) for p in static_dict.values()])


def flame_module(fc_model_list, fc_global_model, model_list, global_model, device):
    pre_global_model = copy.deepcopy(global_model)
    cos = []
    cos_ = []

    for fc_model_out in fc_model_list:
        x1 = vectorize_net(fc_model_out) - vectorize_net(fc_global_model)
        for fc_model_in in fc_model_list:
            x2 = vectorize_net(fc_model_in) - vectorize_net(fc_global_model)
            cos.append(torch.cosine_similarity(x1, x2, dim=0).detach().cpu())
        cos_.append(torch.cat([p.view(-1) for p in cos]).reshape(-1, 1))
        cos = []

    cos_ = torch.cat([p.view(1, -1) for p in cos_])

    cluster_er = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster_er.fit_predict(cos_)
    majority = Counter(cluster_labels)
    res = majority.most_common(len(fc_model_list))

    out = []  # 正常模型列表

    for i in range(len(cluster_labels)):  # 筛选出正常模型
        if cluster_labels[i] == res[0][0]:
            out.append(i)

    e = []  # 每个模型和全局模型的欧氏距离
    for i in range(len(model_list)):
        e.append(
            torch.sqrt(
                torch.sum(
                    (vectorize_net(pre_global_model) -
                     vectorize_net(model_list[i])) ** 2)))

    e = torch.cat([p.view(-1) for p in e])
    st = torch.median(e)
    whole_aggregator = []
    par = []

    for i in range(len(out)):
        par.append(min(1, st / e[out[i]]))

    wa = []
    for p_index, p in enumerate(pre_global_model):
        wa.append(pre_global_model[p])

    for p_index, p in enumerate(model_list[0]):
        # initial
        params_aggregator = torch.zeros(model_list[0][p].size()).to(device)

        for i in range(len(out)):
            net = model_list[out[i]]
            params_aggregator = params_aggregator + \
                                wa[p_index] + (net[p] - wa[p_index]) * par[i]

        sum = 0
        for i in range(len(par)):
            sum += 1
        params_aggregator = params_aggregator / sum
        whole_aggregator.append(params_aggregator)
    lamda = 1e-3
    sigma = st * lamda

    for param_index, p in enumerate(pre_global_model):
        pre_global_model[p] = whole_aggregator[param_index] + \
                              (sigma ** 2) * torch.randn(whole_aggregator[param_index].shape).to(device)

    return pre_global_model


def replace_fc_layers(normal_model, fc_model):
    for fc_key, fc_value in fc_model.items():
        if fc_key in normal_model:
            normal_model[fc_key] = fc_value
    return normal_model


def extract_fc_layers(global_model):
    fc_layers = OrderedDict()
    for key, value in global_model.items():
        if "linear" in key:
            fc_layers.update({key: value})
    return fc_layers


def extract_feature_layers(global_model):
    feature_layers = OrderedDict()
    for key, value in global_model.items():
        if "linear" not in key:
            feature_layers.update({key: value})
    return feature_layers


def create_fc_layers_models(model_list, type_of_model):
    if type_of_model == 'global_model':
        fc_layers_model = extract_fc_layers(model_list)
        return fc_layers_model
    else:
        fc_model_list = []
        for model in model_list:
            fc_layers_model = extract_fc_layers(model)
            fc_model_list.append(fc_layers_model)
        return fc_model_list


def small_flame(model_list, global_model, device, calculate_time):
    global_fc_model = create_fc_layers_models(global_model, "global_model")
    fc_model_list = create_fc_layers_models(model_list, "fc_model_list")

    if calculate_time:
        start_time = time.time()
        _ = flame_module(
            fc_model_list=fc_model_list,
            fc_global_model=global_fc_model,
            global_model=global_model,
            model_list=model_list,
            device=device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"函数调用耗时: {elapsed_time} 秒")  # 函数调用耗时: 0.18109893798828125 秒
        raise SystemExit("error aggregate function!")
    else:
        new_global_model = flame_module(
            fc_model_list=fc_model_list,
            fc_global_model=global_fc_model,
            global_model=global_model,
            model_list=model_list,
            device=device)
        return new_global_model
