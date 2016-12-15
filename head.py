from keras import backend as K
import theano.tensor as T


def batch_reading(head_num, memory_size, memory_dim, memory_t, weight_t):
    """
    Reading memory.
    :param head_num:
    :param memory_size:
    :param memory_dim:
    :param memory_t: the $N \times M$ memory matrix at time $t$, where $N$
    is the number of memory locations, and $M$ is the vector size at each
    location.
    :param weight_t: $w_t$ is a vector of weightings over the $N$ locations
    emitted by a reading head at time $t$.

    Since all weightings are normalized, the $N$ elements $w_t(i)$ of
    $\textbf{w}_t$ obey the following constraints:
    $$\sum_{i=1}^{N} w_t(i) = 1, 0 \le w_t(i) \le 1,\forall i$$

    The length $M$ read vector $r_t$ returned by the head is defined as a
    convex combination of the row-vectors $M_t(i)$ in memory:
    $$\textbf{r}_t \leftarrow \sum_{i=1}^{N}w_t(i)\textbf{M}_t(i)$$
    :return: the content reading from memory.
    """
    r_t_list = K.zeros_like((head_num * memory_dim, 1))

    for i in xrange(head_num):
        begin = i * memory_size
        end = begin + memory_size
        r_t = reading(memory_t, weight_t[begin:end])
        r_t_list[begin:end] = r_t

    return r_t_list


def reading(memory_t, weight_t):
    """
    Reading memory.
    :param memory_t: the $N \times M$ memory matrix at time $t$, where $N$
    is the number of memory locations, and $M$ is the vector size at each
    location.
    :param weight_t: $w_t$ is a vector of weightings over the $N$ locations
    emitted by a reading head at time $t$.

    Since all weightings are normalized, the $N$ elements $w_t(i)$ of
    $\textbf{w}_t$ obey the following constraints:
    $$\sum_{i=1}^{N} w_t(i) = 1, 0 \le w_t(i) \le 1,\forall i$$

    The length $M$ read vector $r_t$ returned by the head is defined as a
    convex combination of the row-vectors $M_t(i)$ in memory:
    $$\textbf{r}_t \leftarrow \sum_{i=1}^{N}w_t(i)\textbf{M}_t(i)$$
    :return: the content reading from memory.
    """
    r_t = K.dot(memory_t, weight_t)
    return r_t


def batch_writing(
        head_num, memory_size, memory_dim, memory_t_1,
        weight_t, eraser_t, adder_t):
    memory_t = memory_t_1

    for i in xrange(head_num):
        # get the addressing for writing
        address_begin = i * memory_size
        address_end = address_begin + memory_size
        content_begin = i * memory_dim
        content_end = content_begin + memory_dim
        memory_t = writing(
            memory_t_1,
            weight_t[address_begin:address_end],
            eraser_t[content_begin:content_end],
            adder_t[content_begin:content_end])
        memory_t_1 = memory_t

    return memory_t


def writing(memory_t_1, weight_t, eraser_t, adder_t):
    """
    Each writing process contain two parts: an erase followed by an add.
    :param memory_t_1: the $N \times M$ memory matrix at time $t-1$, where $N$
    is the number of memory locations, and $M$ is the vector size at each
    location.
    :param weight_t: $w_t$ is a vector of weightings over the $N$ locations
    emitted by a writing head at time $t$.
    :param eraser_t:
    :param adder_t:
    :return:
    """
    # erase
    _memory_t = erase(memory_t_1, weight_t, eraser_t)

    # add
    memory_t = add(_memory_t, weight_t, adder_t)
    return memory_t


def erase(memory_t_1, weight_t, eraser_t):
    '''

    :param memory_t_1:
    :param weight_t:
    :param eraser_t:
    :return:
    '''
    memory = memory_t_1 - T.outer(eraser_t, weight_t)
    # memory = memory_t_1 * (1 - weight_t * eraser_t)
    return memory


def add(_memory_t, weight_t, adder_t):
    '''

    :param _memory_t:
    :param weight_t:
    :param adder_t:
    :return:
    '''
    memory_t = _memory_t + T.outer(adder_t, weight_t)
    # memory_t = _memory_t + weight_t * adder_t
    return memory_t
