


def initial(number_of_memory_locations, vector_size):
    return number_of_memory_locations * vector_size


def addressing(
        memory_t,
        weight_t_1,
        key_vector_t, key_strength_t,
        interpolation_gate_t,
        shift_weight_t,
        scalar_t):
    """
    Addressing mechanisms.
    :param memory_t:
    :param weight_t_1:
    :param key_vector_t:
    :param key_strength_t:
    :param interpolation_gate_t:
    :param shift_weight_t:
    :param scalar_t:
    :return:
    """
    weight_content_t = content_addressing(memory_t, weight_t_1,
                                          key_vector_t, key_strength_t)
    weight_gated_t = interpolation(weight_t_1, weight_content_t, interpolation_gate_t)
    _weight_t = convolutional_shift(weight_gated_t, shift_weight_t)
    weight_t = sharpen(_weight_t, scalar_t)
    return weight_t


def content_addressing(memory_t, weight_t_1, key_vector_t, key_strength_t):
    weight_content_t = 1
    return weight_content_t


def interpolation(weight_t_1, weight_content_t, interpolation_gate_t):
    weight_gated_t = 1
    return weight_gated_t


def convolutional_shift(weight_gated_t, shift_weight_t):
    _weight_t = 1
    return _weight_t


def sharpen(_weight_t, scalar_t):
    weight_t = 1
    return weight_t