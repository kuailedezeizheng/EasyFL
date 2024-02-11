import mxnet as mx
import numpy as np
from copy import deepcopy
import time
from numpy import random
from mxnet import nd, autograd, gluon

def full_trim(v, f):
    '''
    Full-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    # first compute the statistics
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    for i in range(f):
        # apply attack to compromised worker devices with randomness
        random_12 = 2
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v
