import copy
import torch
import hdbscan
from collections import Counter


def flame(global_model_pre, client_model, device, epsilon=3705, lamda=1e-3):
    def vectorize_net(net):
        return torch.cat([p.view(-1) for p in net.parameters()])

    def compute_cosine_similarity(model1, model2):
        x1 = vectorize_net(model1) - vectorize_net(net_avg)
        x2 = vectorize_net(model2) - vectorize_net(net_avg)
        return torch.cosine_similarity(x1, x2, dim=0).detach().cpu()

    net_avg = copy.deepcopy(global_model_pre)
    cos_ = []

    for i in range(len(client_model)):
        cos = [compute_cosine_similarity(client_model[i], client_model[j]) for j in range(len(client_model))]
        cos_.append(torch.cat([p.view(-1) for p in cos]).reshape(-1, 1))

    cos_ = torch.cat([p.view(1, -1) for p in cos_])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = clusterer.fit_predict(cos_)
    majority = Counter(cluster_labels)
    res = majority.most_common(len(client_model))

    out = [i for i in range(len(cluster_labels)) if cluster_labels[i] == res[0][0]]

    e = [torch.sqrt(torch.sum((vectorize_net(net_avg) - vectorize_net(client_model[i])) ** 2))
         for i in range(len(client_model))]
    e = torch.cat([p.view(-1) for p in e])

    st = torch.median(e)

    whole_aggregator = []

    for param_index, p in enumerate(net_avg.parameters()):
        wa = p.data.clone()

        params_aggregator = torch.zeros(p.size()).to(device)

        for i in out:
            net = client_model[i]
            params_aggregator = params_aggregator + wa + (list(net.parameters())[param_index].data - wa) * min(1,
                                                                                                               st / e[
                                                                                                                   i])

        params_aggregator = params_aggregator / len(out)
        whole_aggregator.append(params_aggregator)

    sigma = st * lamda

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index] + (sigma ** 2) * torch.randn(whole_aggregator[param_index].shape).to(
            device)

    return net_avg
