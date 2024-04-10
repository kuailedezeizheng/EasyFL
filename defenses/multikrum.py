import torch
import logging
from collections import OrderedDict


def multikrum(model_weights_list, global_model_weights, root_train_dataset, device, args):
    """Aggregate weight updates from the clients using multi-krum."""
    remaining_weights = flatten_weights(model_weights_list)

    num_attackers_selected = 2
    candidates = []

    # Search for candidates based on distance
    while len(remaining_weights) > 2 * num_attackers_selected + 2:
        distances = []
        for weight in remaining_weights:
            distance = torch.norm((remaining_weights - weight), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(
            distances[:, : len(remaining_weights) - 2 - num_attackers_selected],
            dim=1,
        )
        indices = torch.argsort(scores)
        candidates = (
            remaining_weights[indices[0]][None, :]
            if not len(candidates)
            else torch.cat(
                (candidates, remaining_weights[indices[0]][None, :]), 0
            )
        )

        # Remove candidates from remainings
        remaining_weights = torch.cat(
            (
                remaining_weights[: indices[0]],
                remaining_weights[indices[0] + 1:],
            ),
            0,
        )

    mean_weights = torch.mean(candidates, dim=0)

    # Update global model
    start_index = 0
    mkrum_update = OrderedDict()
    for name, weight_value in model_weights_list[0].items():
        mkrum_update[name] = mean_weights[
                             start_index: start_index + len(weight_value.view(-1))
                             ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished multi-krum server aggregation.")
    return mkrum_update


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
