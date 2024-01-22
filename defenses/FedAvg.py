import copy
import torch


def fed_avg(w_list):
    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg
