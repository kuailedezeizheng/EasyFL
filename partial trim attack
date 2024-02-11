import mxnet as mx
import numpy as np
from copy import deepcopy
import time
from numpy import random
from mxnet import nd, autograd, gluon

def partial_trim(v, f):
    '''
    Partial-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised. 
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    # first compute the statistics
    vi_shape = v[0].shape
    all_grads = nd.concat(*v, dim=1)
    adv_grads = all_grads[:, :f]
    e_mu = nd.mean(adv_grads, axis=1)  # mean
    e_sigma = nd.sqrt(nd.sum(nd.square(nd.subtract(adv_grads, e_mu.reshape(-1, 1))), axis=1) / f)  # standard deviation

    for i in range(f):
        # apply attack to compromised worker devices with randomness
        #norm = nd.norm(v[i])
        v[i] = (e_mu - nd.multiply(e_sigma, nd.sign(e_mu)) * 3.5).reshape(vi_shape)
        #v[i] = v[i]*norm / nd.norm(v[i])

    return v
