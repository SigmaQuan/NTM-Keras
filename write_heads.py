import util


def writing(memory_t, weight_t, eraser_t, adder_t):
    """
    Each writing process contain two parts: an erase followed by an add.
    :param memory_t: the $N \times M$ memory matrix at time $t$, where $N$
    is the number of memory locations, and $M$ is the vector size at each
    location.
    :param weight_t: $w_t$ is a vector of weightings over the $N$ locations
    emitted by a writing head at time $t$.
    :param eraser_t:
    :param adder_t:
    :return:
    """
    memory = add(erase(memory_t, weight_t, eraser_t), weight_t, adder_t)
    return memory


def erase(memory_t, weight_t, eraser_t):
    memory = memory_t * (1 - weight_t * eraser_t)
    return memory


def add(memory_t, weight_t, adder_t):
    memory = memory_t + weight_t * adder_t
    return memory
