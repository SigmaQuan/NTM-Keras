from keras import backend as K
import numpy as np


def initial(number_of_memory_locations, memory_vector_size):
    return K.random_uniform(
        (number_of_memory_locations, memory_vector_size), 0, 1)


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
        memory_t, weight_t_1,
        key_vector_t, key_strength_t)

    # Interpolation
    weight_gated_t = interpolation(
        weight_t_1, weight_content_t, interpolation_gate_t)

    # Convolutional Shift
    _weight_t = convolutional_shift(weight_gated_t, shift_weight_t)

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
    :param memory_t:
    :param key_vector_t:
    :param key_strength_t:
    :return:
    '''
    _weight_content_t = \
        key_strength_t * cosine_similarity(key_vector_t, memory_t)
    weight_content_t = softmax(_weight_content_t)
    return weight_content_t


def interpolation(weight_t_1, weight_content_t, interpolation_gate_t):
    '''
    Focusing by location.
    :param weight_t_1:
    :param weight_content_t:
    :param interpolation_gate_t:
    :return:
    '''
    weight_gated_t = interpolation_gate_t * weight_content_t + \
                     (1 - interpolation_gate_t) * weight_t_1
    return weight_gated_t


def convolutional_shift(weight_gated_t, shift_weight_t):
    '''
    Convolutional shift.
    :param weight_gated_t:
    :param shift_weight_t:
    :return:
    '''
    _weight_t = K.zeros(weight_gated_t.shape)
    for i in np.arange(weight_gated_t.shape[0]):
        for j in np.arnage(shift_weight_t.shape[0]):
            _weight_t[i] += weight_gated_t[j] * shift_weight_t[i - j]
    return _weight_t


def sharpen(_weight_t, scalar_t):
    '''

    :param _weight_t:
    :param scalar_t:
    :return:
    '''
    weight_t = K.pow(_weight_t, scalar_t)
    return weight_t / K.sum(weight_t)
