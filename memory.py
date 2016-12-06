from keras import backend as K
from theano import tensor as T
import numpy as np
import math


def initial(number_of_memory_locations, memory_vector_size):
    return K.zeros(
        (number_of_memory_locations, memory_vector_size), 0, 1)


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


def addressing(
        memory_t,
        weight_t_1,
        key_vector_t, key_strength_t,
        interpolation_gate_t,
        shift_weight_t,
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
    print(weight_content_t)

    # Interpolation
    weight_gated_t = interpolation(
        weight_t_1, weight_content_t, interpolation_gate_t)
    print("weight_content_t")
    print(weight_gated_t)


    # Convolutional Shift
    _weight_t = circular_convolutional_shift(weight_gated_t, shift_weight_t)

    # Sharpening
    weight_t = sharpen(_weight_t, scalar_t)

    return weight_t


def cosine_similarity(u, v):
    return K.dot(u, v) / (K.sum(K.abs(u), axis=1) * K.sum(K.abs(v), axis=1))


def softmax(x):
    return K.softmax(x)


def content_addressing(memory_t,  key_vector_t, key_strength_t):
    '''
    Focusing by content.
    :param memory_t: external memory.
    :param key_vector_t: key vector.
    :param key_strength_t: the strength of key.
    :return:
    '''
    _weight_content_t = \
        key_strength_t * cosine_similarity(key_vector_t, memory_t)
    weight_content_t = softmax(_weight_content_t)
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
                     (1 - interpolation_gate_t) * weight_t_1
    return weight_gated_t

    # shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[
    #         np.arange(-(shift_width // 2), (shift_width // 2) + 1)
    #     ][::-1]
    #
    # def log_shift_convolve(log_weight, log_shift):
    #     # weight: batch_size x mem_size
    #     # shift:  batch_size x shift_width
    #     log_shift = log_shift.dimshuffle(0,1,'x')
    #     log_weight_windows = log_weight[:,shift_conv]
    #     # batch_size x shift_width x mem_size
    #     log_shifted_weight = ops.log_sum_exp(log_shift + log_weight_windows,axis=1)
    #     return log_shifted_weight

#
# def circular_convolution(v, k):
#     """Computes circular convolution.
#
#     Args:
#         v: a 1-D `Tensor` (vector)
#         k: a 1-D `Tensor` (kernel)
#     """
#     size = int(v.get_shape()[0])
#     kernel_size = int(k.get_shape()[0])
#     kernel_shift = int(math.floor(kernel_size/2.0))
#
#     def loop(idx):
#         if idx < 0: return size + idx
#         if idx >= size : return idx - size
#         else: return idx
#
#     kernels = []
#     for i in xrange(size):
#         indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
#         v_ = tf.gather(v, indices)
#         kernels.append(tf.reduce_sum(v_ * k, 0))
#
#     # # code with double loop
#     # for i in xrange(size):
#     #     for j in xrange(kernel_size):
#     #         idx = i + kernel_shift - j + 1
#     #         if idx < 0: idx = idx + size
#     #         if idx >= size: idx = idx - size
#     #         w = tf.gather(v, int(idx)) * tf.gather(kernel, j)
#     #         output = tf.scatter_add(output, [i], tf.reshape(w, [1, -1]))
#
#     return tf.dynamic_stitch([i for i in xrange(size)], kernels)


def circular_convolutional_shift(weight_gated_t, shift_weight_t):
    '''
    Convolutional shift.
    :param weight_gated_t: the weight vector.
    :param shift_weight_t: it defines a normalised distribution over the
    allowed integer shifts (location shift range).
    :return: the shifted weight.
    '''
    print(weight_gated_t)
    print(shift_weight_t)
    print(T.shape(weight_gated_t))
    print(T.shape(shift_weight_t))
    N = T.shape(weight_gated_t)[0]
    shift = 2. * shift_weight_t - 1.
    z = T.mod(shift + N, N)
    shift_t_imj = 1 - (z - T.floor(z))
    # imj = K.mod(K.arange(N) + K.iround(K.floor(z)), N)
    weight_gated_t_roll_1 = T.roll(weight_gated_t, -T.iround(T.floor(z)))
    weight_gated_t_roll_2 = T.roll(weight_gated_t, -(T.iround(T.floor(z)) + 1))
    _weight_t = weight_gated_t_roll_1 * shift_t_imj + \
                weight_gated_t_roll_2 * (1 - shift_t_imj)

    return _weight_t

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

