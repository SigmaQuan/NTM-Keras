import numpy as np
import random


np.random.seed(7883)
random.seed(7883)


def generate_random_binomial(row, col):
    return np.random.binomial(
        1, 0.5, (row, col)).astype(np.uint8)


def generate_weightings(row, col):
    write_weightings = np.zeros((row, col), dtype=np.float32)
    read_weightings = np.zeros((row, col), dtype=np.float32)
    r = (row * 3) / 4
    for i in np.arange(0, col/2):
        write_weightings[r][i] = 1
        read_weightings[r][i + col/2] = 1
        r = r - 1

    return write_weightings, read_weightings


def generate_copy_data(input_size, sequence_length):
    # produce random sequence
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)

    # allocate space for input sequence and output sequence
    input_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.bool)

    # set value of input sequence
    input_sequence[:sequence_length, :-1] = sequence
    # "1": A special flag which indicate the end of the input
    input_sequence[sequence_length, -1] = 1

    # set value of output sequence
    output_sequence[sequence_length + 1:, :-1] = sequence
    # "1": A special flag which indicate the begin of the output
    output_sequence[sequence_length, -1] = 1

    return input_sequence, output_sequence


def generate_copy_data_set(
        input_dimension,
        max_length_of_original_sequence,
        example_size):
    # get random sequence lengths from uniform distribution [1, 20]
    sequence_lengths = np.random.randint(
        1, max_length_of_original_sequence + 1, example_size)

    # allocate space for input sequences and output sequences, where the
    # "1" is a special flag which indicate the end of the input or output
    input_sequences = np.zeros(
        (example_size, max_length_of_original_sequence * 2 + 1, input_dimension),
        dtype=np.bool)
    output_sequences = np.zeros(
        (example_size, max_length_of_original_sequence * 2 + 1, input_dimension),
        dtype=np.bool)

    # set the value for input sequences and output sequences
    for i in range(example_size):
        input_sequence, output_sequence = generate_copy_data(
            input_dimension, sequence_lengths[i])
        input_sequences[i, :sequence_lengths[i]*2+1] = input_sequence
        output_sequences[i, :sequence_lengths[i]*2+1] = output_sequence

    return input_sequences, output_sequences


def generate_repeat_copy_data(input_size, sequence_length, repeat_times):
    # produce random sequence
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)

    # allocate space for input sequence and output sequence
    input_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * repeat_times,  # + 1
         input_size),
        dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * repeat_times,  # + 1
         input_size),
        dtype=np.bool)

    # set value of input sequence
    input_sequence[:sequence_length, :-1] = sequence
    # input_sequence[sequence_length, -1] = repeat_times
    input_sequence[sequence_length, -1] = 1

    # set value of output sequence  ## sequence_length + 1
    output_sequence[sequence_length+1:, :-1] = \
        np.tile(sequence, (repeat_times, 1))
    # "1": A special flag which indicate the begin of the output
    # output_sequence[sequence_length, -1] = 1

    return input_sequence, output_sequence


def generate_repeat_copy_data_set(
        input_dimension,
        max_length_of_original_sequence,
        example_size,
        max_repeat_times):
    # produce random sequence lengths from uniform distribution
    # [1, max_length]
    sequence_lengths = np.random.randint(
        1, max_length_of_original_sequence + 1, example_size)

    # produce random repeat times from uniform distribution
    # [1, max_repeat_times]
    repeat_times = np.random.randint(1, max_repeat_times + 1, example_size)
    input_sequences = np.zeros(
        (example_size,
         max_length_of_original_sequence * (max_repeat_times + 1) + 1,  # + 1
         input_dimension),
        dtype=np.bool)
    output_sequences = np.zeros(
        (example_size,
         max_length_of_original_sequence * (max_repeat_times + 1) + 1,  # + 1
         input_dimension),
        dtype=np.bool)

    # set the value for input sequences and output sequences
    for i in range(example_size):
        input_sequence, output_sequence = generate_repeat_copy_data(
            input_dimension, sequence_lengths[i], repeat_times[i])
        input_sequences[i, :sequence_lengths[i]*(repeat_times[i]+1)+1] = \
            input_sequence
        output_sequences[i, :sequence_lengths[i]*(repeat_times[i]+1)+1] = \
            output_sequence

    return input_sequences, output_sequences, repeat_times


def generate_associative_recall_items(input_size, item_size, episode_size):
    inner_item = np.random.binomial(
        1, 0.5, ((item_size + 1) * episode_size, input_size)
    ).astype(np.uint8)
    items = np.zeros(((item_size + 1) * episode_size, input_size + 2),
                     dtype=np.uint8)
    # items = np.zeros(((item_size + 1) * episode_size, input_size + 2),
    #                  dtype=np.bool)
    items[:, :-2] = inner_item

    separator = np.zeros((1, input_size + 2), dtype=np.uint8)
    # separator = np.zeros((1, input_size + 2), dtype=np.bool)
    separator[0][-2] = 1
    items[:(item_size + 1) * episode_size:(item_size + 1)] = separator[0]

    return items


def generate_associative_recall_data(
        input_size, item_size, episode_size, max_episode_size):
    sequence_length = (item_size+1) * (max_episode_size+2)
    input_sequence = np.zeros(
        (sequence_length, input_size + 2), dtype=np.uint8)
    # input_sequence = np.zeros(
    #     (sequence_length, input_size + 2), dtype=np.bool)
    input_sequence[:(item_size + 1) * episode_size] = \
        generate_associative_recall_items(
            input_size, item_size, episode_size)

    separator = np.zeros((1, input_size+2), dtype=np.uint8)
    # separator = np.zeros((1, input_size + 2), dtype=np.bool)
    separator[0][-2] = 1
    query_index = np.random.randint(0, episode_size-1)

    input_sequence[(item_size+1)*episode_size:(item_size+1)*(episode_size+1)] = \
        input_sequence[(item_size+1)*query_index:(item_size+1)*(query_index+1)]
    input_sequence[(item_size+1)*episode_size][-2] = 0
    input_sequence[(item_size+1)*episode_size][-1] = 1
    input_sequence[(item_size+1)*(episode_size+1)][-1] = 1

    output_sequence = np.zeros(
        (sequence_length, input_size + 2), dtype=np.uint8)
    # output_sequence = np.zeros(
    #     (sequence_length, input_size + 2), dtype=np.bool)
    output_sequence[(item_size+1)*(episode_size+1):(item_size+1)*(episode_size+2)] = \
        input_sequence[(item_size+1)*(query_index+1):(item_size+1)*(query_index+2)]
    output_sequence[(item_size+1)*(episode_size+1)][-2] = 0

    return input_sequence, output_sequence


def generate_associative_recall_data_set(
        input_size, item_size, max_episode_size, example_size):
    episode_size = np.random.randint(2, max_episode_size + 1, example_size)
    sequence_length = (item_size+1) * (max_episode_size+2)
    input_sequences = np.zeros((example_size, sequence_length, input_size + 2), dtype=np.uint8)
    output_sequences = np.zeros((example_size, sequence_length, input_size + 2), dtype=np.uint8)
    # input_sequences = np.zeros((training_size, sequence_length, input_size + 2), dtype=np.bool)
    # output_sequences = np.zeros((training_size, sequence_length, input_size + 2), dtype=np.bool)
    for i in range(example_size):
        input_sequence, output_sequence = generate_associative_recall_data(
            input_size, item_size, episode_size[i], max_episode_size)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence

    return input_sequences, output_sequences


def generate_probability_of_n_gram_by_beta(a, b, n):
    return np.random.beta(a, b, np.power(2, n-1))


def get_index(n_1_bits, n):
    index = n_1_bits[0]
    for i in range(1, n-1):
        index = index + np.power(2, i) * n_1_bits[i]

    return index


def generate_dynamical_n_gram_data(look_up_table, n, sequence_length):
    example_number = 100
    input_size = 1
    input_sequence = np.zeros((example_number, sequence_length*2-n+2, input_size+2), dtype=np.uint8)
    output_sequence = np.zeros((example_number, sequence_length*2-n+2, input_size+2), dtype=np.uint8)

    input_sequence_ = np.zeros((sequence_length*2-n+2, input_size+2), dtype=np.uint8)
    output_sequence_ = np.zeros((sequence_length*2-n+2, input_size+2), dtype=np.uint8)
    input_sequence_[0:n-1, 0] = np.random.binomial(1, 0.5, (1, n-1)).astype(np.uint8)
    # for i in range(n-1, sequence_length):
    #     n_1_bits = input_sequence[i-n+1: i]
    #     index = get_index(n_1_bits, n)
    #     input_sequence[i] = np.random.binomial(1, look_up_table[index], 1)
    # output_sequence[n-1: -1] = input_sequence[n-1: -1]

    for i in range(n-1, sequence_length):
        n_1_bits = input_sequence_[i-n+1: i, 0]
        index = get_index(n_1_bits, n)
        # input_sequence_[i][0] = np.random.binomial(1, look_up_table[index], 1)
        # output_sequence_[sequence_length+i-n+2][0] = np.random.binomial(1, look_up_table[index], 1)
        input_sequence[:, i, 0] = np.random.binomial(1, look_up_table[index], 1)
        # output_sequence_[sequence_length+i-n+2][0] = np.random.binomial(1, look_up_table[index], 1)
        output_sequence[:, sequence_length+i-n+2, 0] = np.random.binomial(
            1, look_up_table[index], example_number)
    input_sequence[:, sequence_length, -1] = 1
    input_ones = np.ones((example_number, sequence_length))
    input_sequence[:, 0:sequence_length, 1] = \
        input_ones - input_sequence[:, 0:sequence_length, 0]
    output_ones = np.ones((example_number, sequence_length-n+1))
    output_sequence[:, sequence_length+1:sequence_length*2-n+2, 1] = \
        output_ones - output_sequence[:, sequence_length+1:sequence_length*2-n+2, 0]

    # print(input_sequence_.shape)
    # input_sequence_[0:sequence_length, 0] = input_sequence
    # input_sequence_[sequence_length, -1] = 1
    # output_sequence_[1, sequence_length+1:sequence_length*2-n+2] = input_sequence

    # print(input_sequence)
    # print(output_sequence)

    return input_sequence, output_sequence


def generate_dynamical_n_gram_data_set(
        look_up_table, n, sequence_length, example_size):
    input_size = 1
    input_sequences = np.zeros((example_size, sequence_length*2-n+2, input_size+2), dtype=np.uint8)
    output_sequences = np.zeros((example_size, sequence_length*2-n+2, input_size+2), dtype=np.uint8)
    # input_sequences = np.zeros((example_size, sequence_length, input_size), dtype=np.uint8)
    # output_sequences = np.zeros((example_size, sequence_length, input_size), dtype=np.uint8)
    # input_sequences = np.zeros((example_size, sequence_length, 1), dtype=np.bool)
    # output_sequences = np.zeros((example_size, sequence_length, 1), dtype=np.bool)
    for i in range(example_size/100):
        input_sequence, output_sequence = generate_dynamical_n_gram_data(
            look_up_table, n, sequence_length)
        input_sequences[i*100:(i+1)*100] = input_sequence
        output_sequences[i*100:(i+1)*100] = output_sequence
        # print(i)
        # print(input_sequence)
        # print(output_sequence)

    return input_sequences, output_sequences


def generate_associative_priority_sort_data(
        input_size,
        input_sequence_length,
        output_sequence_length,
        priority_lower_bound,
        priority_upper_bound):
    sequence = input_sequence_length + output_sequence_length + 1
    input_sequence = np.random.binomial(
        1, 0.5, (input_sequence_length, input_size+1)).astype(np.uint8)
    output_sequence = np.zeros(
        (output_sequence_length, input_size+1), dtype=np.uint8)
    input_priority = np.random.uniform(priority_lower_bound,
                                       priority_upper_bound,
                                       (input_sequence_length, 1))
    output_priority = sorted(
        input_priority, reverse=True)[:output_sequence_length]
    pair = [(input_sequence[i], input_priority[i])
            for i in range(input_sequence_length)]
    sorted_input_sequence = sorted(
        pair, key=lambda prior: prior[1], reverse=True)
    for i in range(output_sequence_length):
        output_sequence[i] = sorted_input_sequence[i][0]

    input_sequence_ = np.zeros((sequence, input_size+2), dtype=np.float32)
    input_priority_ = np.zeros((sequence, 1), dtype=np.float32)
    output_sequence_ = np.zeros((sequence, input_size+2), dtype=np.float32)
    output_priority_ = np.zeros((sequence, 1), dtype=np.float32)

    input_sequence_[:input_sequence_length, :-1] = input_sequence
    input_sequence_[input_sequence_length][-1] = 1
    input_priority_[:input_sequence_length] = input_priority
    output_sequence_[input_sequence_length+1:sequence, :-1] = output_sequence
    output_priority_[input_sequence_length+1:sequence] = output_priority

    # return input_sequence, input_priority, output_sequence, output_priority
    return input_sequence_, input_priority_, output_sequence_, output_priority_


def generate_associative_priority_sort_data_set(
        input_size,
        input_sequence_length,
        output_sequence_length,
        priority_lower_bound,
        priority_upper_bound,
        example_size):
    sequence_length = input_sequence_length + output_sequence_length
    input_sequences = np.zeros((example_size, sequence_length+1, input_size+2), dtype=np.float32)
    output_sequences = np.zeros((example_size, sequence_length+1, input_size+2), dtype=np.float32)
    for i in range(example_size):
        input_sequence, input_priority, output_sequence, output_priority = \
            generate_associative_priority_sort_data(
                input_size,
                input_sequence_length,
                output_sequence_length,
                priority_lower_bound,
                priority_upper_bound)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence
        input_sequences[i][:, -2] = input_priority.transpose()
        output_sequences[i][:, -2] = output_priority.transpose()

    return input_sequences, output_sequences
