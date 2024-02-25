import torch
import logging
from plato.config import Config
from scipy.stats import norm
from collections import OrderedDict
import numpy as np
import os

def oblivion_min_max_attack(weights_received, dev_type="unit_vec"):
    """
    Attack name: Min-max with Oblivion

    """

    attacker_weights = flatten_weights(weights_received)

    weights_avg = torch.mean(attacker_weights, 0)

    # Smooth benign model updates
    weights_avg = smoothing("benign", weights_avg)

    # Generate perturbation vectors (Inverse unit vector by default)
    if dev_type == "unit_vec":
        # Inverse unit vector
        perturbation_vector = weights_avg / torch.norm(weights_avg)
    elif dev_type == "sign":
        # Inverse sign
        perturbation_vector = torch.sign(weights_avg)
    elif dev_type == "std":
        # Inverse standard deviation
        perturbation_vector = torch.std(attacker_weights, 0)

    # Importance pruning
    sali_indicators_vector = compute_sali_indicator()
    perturbation_vector = perturbation_vector * sali_indicators_vector

    # Calculate the maximum distance between any two benign updates (unpoisoned)
    max_distance = torch.tensor([0])
    for attacker_weight in attacker_weights:
        distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
        max_distance = torch.max(max_distance, torch.max(distance))

    # Search for lambda such that its maximum distance from any other gradient is bounded
    lambda_value = torch.Tensor([50.0]).float()
    threshold = 1e-5
    lambda_step = lambda_value
    lambda_succ = 0

    while torch.abs(lambda_succ - lambda_value) > threshold:
        poison_value = weights_avg - lambda_value * perturbation_vector
        distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lambda_succ = lambda_value
            lambda_value = lambda_value + lambda_step / 2
        else:
            lambda_value = lambda_value - lambda_step / 2

        lambda_step = lambda_step / 2

    poison_value = weights_avg - lambda_succ * perturbation_vector

    # Smooth poison value
    poison_value = smoothing("poisoned", poison_value)

    # Perform model poisoning
    weights_poisoned = perform_model_poisoning(weights_received, poison_value)
    logging.info(f"Finished Min-Max model poisoning attack (with Oblivion).")
    return weights_poisoned

def perform_model_poisoning(weights_received, poison_value):
    # Poison the reveiced weights based on calculated poison value.
    weights_poisoned = []
    for weight_received in weights_received:
        start_index = 0
        weight_poisoned = OrderedDict()

        for name, weight in weight_received.items():
            weight_poisoned[name] = poison_value[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_poisoned.append(weight_poisoned)
    return weights_poisoned


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
