import copy

import torch


def federated_averaging(model_weights_list, global_model_weights, eta=5, n=100):
        w_avg = copy.deepcopy(model_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(model_weights_list)):
                w_avg[k] += model_weights_list[i][k]
            w_avg[k] = torch.mul(w_avg[k], eta/n) + torch.mul(global_model_weights[k], 1 - eta * len(model_weights_list) / n)
        return w_avg
