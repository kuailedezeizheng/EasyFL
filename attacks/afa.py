import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict
import numpy as np
import copy
from typing import Mapping
import os
import pickle

def afa(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using afa."""

    flattened_weights = flatten_weights(weights_attacked)
    clients_id = [update.client_id for update in updates]

    retrive_flattened_weights = flattened_weights.clone()

    bad_set = []
    remove_set = [1]
    pvalue = {}
    epsilon = 2
    delta_ep = 0.5

    # Load from the history or create new ones
    file_path = "./parameters.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            good_hist = pickle.load(file)
            bad_hist = pickle.load(file)
            alpha = pickle.load(file)
            beta = pickle.load(file)
    else:
        good_hist = np.zeros(Config().clients.total_clients)
        bad_hist = np.zeros(Config().clients.total_clients)
        alpha = 3
        beta = 3

    for counter, client in enumerate(clients_id):
        ngood = good_hist[client - 1]
        nbad = bad_hist[client - 1]
        alpha = alpha + ngood
        beta = beta + nbad
        pvalue[counter] = alpha / (alpha + beta)

    # Search for bad guys
    while len(remove_set):
        remove_set = []

        cos_sims = []
        for weight in flattened_weights:
            cos_sim = (
                torch.dot(weight.squeeze(), final_update.squeeze())
                / (torch.norm(final_update.squeeze()) + 1e-9)
                / (torch.norm(weight.squeeze()) + 1e-9)
            )
            cos_sims = (
                cos_sim.unsqueeze(0)
                if not len(cos_sims)
                else torch.cat((cos_sims, cos_sim.unsqueeze(0)))
            )

        model_mean = torch.mean(cos_sims, dim=0).squeeze()
        model_median = torch.median(cos_sims, dim=0)[0].squeeze()
        model_std = torch.std(cos_sims, dim=0).squeeze()

        flattened_weights_copy = copy.deepcopy(flattened_weights)

        if model_mean < model_median:
            for counter, weight in enumerate(flattened_weights):
                if cos_sims[counter] < (model_median - epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        afa_index_finder(
                            weight, retrive_flattened_weights[counter:]
                        )
                        + counter
                    )
                    delete_id = afa_index_finder(weight, flattened_weights_copy)
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )
                    bad_set.append(remove_id)
                
        else:
            for counter, weight in enumerate(flattened_weights):  #  we for loop this
                if cos_sims[counter] > (model_median + epsilon * model_std):
                    remove_set.append(1)
                    remove_id = (
                        afa_index_finder(
                            weight, retrive_flattened_weights[counter:]
                        )
                        + counter
                    )
                    delete_id = afa_index_finder(weight, flattened_weights_copy)
                    temp_tensor1 = flattened_weights_copy[0:delete_id]
                    temp_tensor2 = flattened_weights_copy[delete_id + 1 :]
                    flattened_weights_copy = torch.cat(
                        (temp_tensor1, temp_tensor2), dim=0
                    )  # but we changes it in the loop, maybe we should get a copy
                    bad_set.append(remove_id)

        epsilon += delta_ep
        flattened_weights = copy.deepcopy(flattened_weights_copy)

    # Update good_hist and bad_hist according to bad_set
    good_set = copy.deepcopy(clients_id)

    # Update history
    for rm_id in bad_set:
        bad_hist[clients_id[rm_id] - 1] += 1
        good_set.remove(clients_id[rm_id])
    for gd_id in good_set:
        good_hist[gd_id - 1] += 1
    with open(file_path, "wb") as file:
        pickle.dump(good_hist, file)
        pickle.dump(bad_hist, file)
        pickle.dump(alpha, file)
        pickle.dump(beta, file)

    # Perform aggregation
    p_sum = 0
    final_update = torch.zeros(flattened_weights[0].shape)
    
    for counter, weight in enumerate(flattened_weights):
        tmp = afa_index_finder(weight, retrive_flattened_weights[counter:])
        if tmp != -1:
            index_value = tmp + counter
            p_sum += pvalue[index_value]
            final_update += pvalue[index_value] * weight

    final_update = final_update / p_sum

    # Update globel weights
    start_index = 0
    afa_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        afa_update[name] = final_update[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished AFA server aggregation.")
    return afa_update

def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights
