from collections import OrderedDict

import torch

from defenses.fed_avg import federated_averaging
from defenses.flame import flame


def vectorize_net(static_dict):
    return torch.cat([p.view(-1) for p in static_dict.values()])


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


def small_flame(model_list, global_model, device):
    global_fc_model = create_fc_layers_models(global_model, "global_model")
    fc_model_list = create_fc_layers_models(model_list, "fc_model_list")
    fc_avg = flame(fc_model_list, global_fc_model, device)
    global_avg_model = federated_averaging(global_weight=global_model, w_list=model_list)
    new_global_model = replace_fc_layers(global_avg_model, fc_avg)
    return new_global_model
