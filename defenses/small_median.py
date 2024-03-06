def small_median(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)
            # distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
            # nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        # distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()

        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    if len(param_list) % 2 == 1:
        median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2]
    else:
        median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2: len(param_list) // 2 + 1].mean(
            axis=-1, keepdims=1)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * median_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return median_nd, distance