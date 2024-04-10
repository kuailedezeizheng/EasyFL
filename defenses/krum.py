import copy
import logging

import torch


def krum(model_weights_list, global_model_weights, root_train_dataset, device, args):
    """Aggregate weight updates from the clients using multi-krum."""
    flatten_models_weights = flatten_weights(model_weights_list)

    num_attackers_selected = 2

    distances = []
    for weight in flatten_models_weights:
        distance = torch.norm((flatten_models_weights - weight), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances = torch.sort(distances, dim=1)[0]

    scores = torch.sum(
        distances[:, : len(flatten_models_weights) - 2 - num_attackers_selected],
        dim=1,
    )
    indices = torch.argsort(scores)

    krum_update = copy.deepcopy(model_weights_list[indices[0]])
    logging.info(f"Finished multi-krum server aggregation.")
    return krum_update


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
