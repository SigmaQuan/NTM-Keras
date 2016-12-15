from keras import backend as K
from theano import tensor as T
import numpy as np
import math
import theano


def initial(number_of_memory_locations, memory_vector_size):
    return K.zeros((number_of_memory_locations, memory_vector_size))


def batch_addressing(
        head_num,
        memory_size,
        memory_t,
        weight_t_1,
        key_vector_t,
        key_strength_t,
        interpolation_gate_t,
        shift_weight_t,
        scalar_t):
    """
    Addressing mechanisms.
    :param head_num: the number of heads.
    :param memory_size:
    :param memory_t: memory matrix at time t.
    :param weight_t_1: memory weight at time t-1.
    :param key_vector_t: key vector at time t.
    :param key_strength_t: strength of key vector at time t.
    :param interpolation_gate_t: interpolation gate at time t.
    :param shift_weight_t: shift weight at time t.
    :param scalar_t: scalar at time t.
    :return: a weight vector at time t.
    """
    w_w_t = K.zeros_like(weight_t_1)
    for i in xrange(head_num):
        # get the addressing for writing
        begin = i * memory_size
        end = begin + memory_size
        w_w_t_i = addressing(
            memory_t,
            weight_t_1[begin:end],
            key_vector_t[begin:end],
            key_strength_t[begin:end],
            interpolation_gate_t[begin:end],
            shift_weight_t[begin:end],
            scalar_t[begin:end])
        w_w_t[begin:end] = w_w_t_i

    return w_w_t

#
# def addressing(
#         memory_t,
#         weight_t_1,
#         key_vector_t, key_strength_t,
#         interpolation_gate_t,
#         shift_weight_t,
#         scalar_t):
#     """
#     Addressing mechanisms.
#     :param memory_t: memory matrix at time t.
#     :param weight_t_1: memory weight at time t-1.
#     :param key_vector_t: key vector at time t.
#     :param key_strength_t: strength of key vector at time t.
#     :param interpolation_gate_t: interpolation gate at time t.
#     :param shift_weight_t: shift weight at time t.
#     :param scalar_t: scalar at time t.
#     :return: a weight vector at time t.
#     """
#     # Content addressing
#     weight_content_t = content_addressing(
#         memory_t, key_vector_t, key_strength_t)
#     print("weight_content_t")
#     print(weight_content_t)
#
#     # Interpolation
#     weight_gated_t = interpolation(
#         weight_t_1, weight_content_t, interpolation_gate_t)
#     print("weight_content_t")
#     print(weight_gated_t)
#
#
#     # Convolutional Shift
#     _weight_t = circular_convolutional_shift(weight_gated_t, shift_weight_t)
#
#     # Sharpening
#     weight_t = sharpen(_weight_t, scalar_t)
#
#     return weight_t


def addressing(
        memory_t,
        memory_dim,
        memory_size,
        weight_t_1,
        key_vector_t, key_strength_t,
        interpolation_gate_t,
        shift_weight_t,
        shift_range,
        scalar_t):
    """
    Addressing mechanisms.
    :param memory_t: memory matrix at time t.
    :param weight_t_1: memory weight at time t-1.
    :param key_vector_t: key vector at time t.
    :param key_strength_t: strength of key vector at time t.
    :param interpolation_gate_t: interpolation gate at time t.
    :param shift_weight_t: shift weight at time t.
    :param scalar_t: scalar at time t.
    :return: a weight vector at time t.
    """
    # Content addressing
    weight_content_t = content_addressing(
        memory_t, key_vector_t, key_strength_t)
    print("weight_content_t")
    # print(weight_content_t)

    # Interpolation
    weight_gated_t = interpolation(
        weight_t_1, weight_content_t, interpolation_gate_t)
    print("weight_gated_t")
    # print(weight_gated_t)

    # Convolutional Shift
    _weight_t = circular_convolutional_shift(
        weight_gated_t, shift_weight_t, memory_size, shift_range)
    print("_weight_t")
    # print(_weight_t)

    # Sharpening
    weight_t = sharpen(_weight_t, scalar_t)
    print("weight_t")
    # print(weight_t)

    return weight_t


def cosine_similarity_group(u, V):
    similairty = K.dot(u, V) / (K.sum(K.abs(u)) * K.sum(K.abs(V), axis=0))
    # import numpy as np
    # u = np.random.random((3))
    # V = np.random.random((3, 4))
    # sim = np.dot(u, V) / (sum(abs(u)) * np.sum(abs(V), axis=0))
    # print("u")
    # print(u)
    # print("V")
    # print(V)
    # print("similairty")
    # print(similairty)
    return similairty


def cosine_similarity(u, v):
    similairty = K.dot(u, v) / (K.sum(K.abs(u)) * K.sum(K.abs(v), axis=0))
    # similairty = K.dot(u, v) / (K.sum(K.abs(u), axis=1) * K.sum(K.abs(v), axis=1))
    # print("u")
    # print(u)
    # print("v")
    # print(v)
    # print("similairty")
    # print(similairty)
    return similairty


def softmax(x):
    # print("x")
    # print(x)
    _softmax = K.softmax(x)
    # print("softmax(x)")
    # print(_softmax)
    return _softmax


def content_addressing(memory_t,  key_vector_t, key_strength_t):
    '''
    Focusing by content.
    :param memory_t: external memory.
    :param key_vector_t: key vector.
    :param key_strength_t: the strength of key.
    :return:
    '''
    # print("content addressing:")
    # print(">>memory_t")
    # print(key_vector_t)
    # print(">>key_vector_t")
    # print(key_vector_t)
    # print(">>key_strength_t")
    # print(key_strength_t)
    _weight_content_t = \
        key_strength_t * cosine_similarity_group(key_vector_t, memory_t)
    weight_content_t = softmax(_weight_content_t)
    # print("_weight_content_t")
    # print(_weight_content_t)
    return weight_content_t


def interpolation(weight_t_1, weight_content_t, interpolation_gate_t):
    '''
    Focusing by location.
    :param weight_t_1: the weight value at time-step t-1
    :param weight_content_t: the weight get by content-based addressing.
    :param interpolation_gate_t: the interpolation gate.
    :return:
    '''
    weight_gated_t = interpolation_gate_t * weight_content_t + \
                     (1.0 - interpolation_gate_t) * weight_t_1
    return weight_gated_t


def circular_convolutional_shift(v, k, n, m):
    """Computes circular convolution.
    Args:
        v: a 1-D `Tensor` (vector)
        k: a 1-D `Tensor` (kernel)
    """
    # size = int(v.get_shape()[0])
    # kernel_size = int(k.get_shape()[0])
    # kernel_shift = int(math.floor(kernel_size/2.0))
    size = n
    kernel_size = m
    kernel_shift = (kernel_size + 1)/2.0
    shift_range = T.argmax(k) - kernel_shift

    def loop(idx):
        if T.lt(idx, 0):
            return size + idx
        if T.ge(idx, size):
            return idx - size
        else:
            return idx

    kernels = []
    for i in T.xrange(size):
        indices = loop(i + shift_range)
        index = theano.tensor.cast(indices, 'int64')
        v_ = v[index]
        kernels.append(v_)

    return kernels

# def circular_convolutional_shift(v, k, n, m):
#     """Computes circular convolution.
#     Args:
#         v: a 1-D `Tensor` (vector)
#         k: a 1-D `Tensor` (kernel)
#     """
#     # size = int(v.get_shape()[0])
#     # kernel_size = int(k.get_shape()[0])
#     # kernel_shift = int(math.floor(kernel_size/2.0))
#     size = n
#     kernel_size = m
#     kernel_shift = (kernel_size + 1)/2.0
#
#     # def loop(idx):
#     #     if idx < 0:
#     #         return size + idx
#     #     if idx >= size:
#     #         return idx - size
#     #     else:
#     #         return idx
#
#     def loop(idx):
#         if idx < 0:
#             return size + idx
#         if T.ge(idx, size):
#             return idx - size
#         else:
#             return idx
#
#     kernels = []
#     # range_list = T.xrange(kernel_shift, -kernel_shift-1, -1)
#     # range_list = theano.tensor.arange(kernel_shift, -kernel_shift-1, -1)
#     #
#     # range_list_, updates_ = theano.scan(lambda i, d: T.sub(m, i), sequences=k)
#     # range_list = theano.function(inputs=[m, k], outputs=range_list_)
#     #
#
#     my_range_max = T.iscalar('my_range_max')
#     my_range = T.arange(my_range_max)
#     get_range_list = theano.function(inputs=[my_range_max], outputs=my_range)
#     range_list = get_range_list(kernel_size)
#
#     # range_list = T.arange(m)
#
#     for i in T.xrange(size):
#         results, updates = theano.scan(lambda r: loop(T.add(r, i)), sequences=range_list)
#         indices = theano.function(inputs=[i, range_list], outputs=results)
#
#         v_ = T.gather(v, indices)
#         kernels.append(T.reduce_sum(v_ * k, 0))
#
#     return T.dynamic_stitch([i for i in T.xrange(size)], kernels)


def sharpen(_weight_t, scalar_gama_t):
    '''
    The convolution operation in convolutional shift can cause leakage or
    dispersion of weights over time if the shift weighting is no sharp.
    For example, if shift of -1, 0 and 1 are given weights of 0.1, 0.8,
    and 0.1, the rotation will transform a weighting focused at single
    point into one slightly blurred over three points. To combat this,
    each head emits one further scalar \gama >= 1 whose effect is sharpen
    the final weighting as follows:
    $$w_{i}^{(t)} = \frac{(\hat{w}_{i}^{(t)})^{\gama}}
    {\sum_{j}\hat{w}_{j}^{(t)})^{\gama}}$$
    :param _weight_t: the weight vector which denotes a memory address.
    :param scalar_gama_t: the scalar for sharpen.
    :return: the sharpened weight.
    '''
    weight_t = K.pow(_weight_t, scalar_gama_t)
    return weight_t / K.sum(weight_t)

