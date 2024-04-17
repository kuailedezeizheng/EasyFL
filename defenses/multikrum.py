from collections import OrderedDict
import torch

from decorators.timing import record_time


def get_gradient_update(user_model_weights, global_model_weights):
    for key, value in user_model_weights.items():
        value = value - global_model_weights[key]
        user_model_weights[key] = value
    return user_model_weights


def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor


def euclidean_distance(x, y):
    return torch.norm(x - y)


@record_time
def multikrum(model_weights_list, global_model_weights, device):
    num_models = len(model_weights_list)
    dist_matrix = torch.zeros(num_models, num_models)
    num_attackers = 10
    num_selected_users = 10
    num_selected_users_float = torch.tensor(num_selected_users, dtype=torch.float32)

    # 计算梯度更新
    for i, user_model_weights in enumerate(model_weights_list):
        model_weights_list[i] = get_gradient_update(user_model_weights, global_model_weights)

    # 计算权重之间的距离
    for i, model_weight_x in enumerate(model_weights_list):
        vectorized_weight_x = vectorize_net(model_weight_x)
        for j in range(i + 1, num_models):
            vectorized_weight_y = vectorize_net(model_weights_list[j])
            dist_matrix[i, j] = euclidean_distance(vectorized_weight_x, vectorized_weight_y)
            dist_matrix[j, i] = euclidean_distance(vectorized_weight_x, vectorized_weight_y)

    # 计算每个参与者的距离和，并选择距离和最小的模型
    user_scores = torch.zeros(num_models)
    for i in range(num_models):
        sorted_indices = torch.argsort(dist_matrix[i])
        sum_dist = torch.sum(dist_matrix[i, sorted_indices[1:(num_models - num_attackers)]])
        user_scores[i] = sum_dist

    user_scores_sorted_indices = torch.argsort(user_scores)
    good_model_indices = user_scores_sorted_indices[:num_selected_users]

    good_model_list = [model_weights_list[i] for i in good_model_indices]

    for key in global_model_weights.keys():
        total_weight = global_model_weights[key].clone().zero_()  # 初始化为0
        for model_weight in good_model_list:
            total_weight += model_weight[key]
        avg_update = total_weight.div(num_selected_users_float).to(global_model_weights[key].dtype)
        global_model_weights[key] += avg_update

    return global_model_weights
