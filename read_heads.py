from keras import backend as K


def reading(head_num, memory_size, memory_dim, memory_t, weight_t):
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
