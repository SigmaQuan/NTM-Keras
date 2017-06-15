# -*- coding: utf-8 -*-
import numpy as np
from utils import initialize_random_seed


# Initialize the random seed
initialize_random_seed()


def generate_weightings(row, col):
    write_weightings = np.zeros((row, col), dtype=np.float32)
    read_weightings = np.zeros((row, col), dtype=np.float32)
    r = (row * 3) / 4
    for i in np.arange(0, col/2):
        write_weightings[r][i] = 1
        read_weightings[r][i + col/2] = 1
        r -= 1

    return write_weightings, read_weightings


def generate_copy_sample(dimension, sequence_length):
    """Generate one sample of copy algorithm.

    # Arguments
        dimension: the dimension of each input output tokens.
        sequence_length: the length of input sequence, i.e. the number of
            input tokens.

    # Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    # produce random sequence
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, dimension - 1)).astype(np.uint8)

    # allocate space for input sequence and output sequence
    input_sequence = np.zeros(
        (sequence_length * 2 + 1, dimension), dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length * 2 + 1, dimension), dtype=np.bool)

    # set value of input sequence
    input_sequence[:sequence_length, :-1] = sequence
    # "1": A special flag which indicate the end of the input
    input_sequence[sequence_length, -1] = 1

    # set value of output sequence
    output_sequence[sequence_length + 1:, :-1] = sequence
    # "1": A special flag which indicate the begin of the output
    output_sequence[sequence_length, -1] = 1

    # return the sample
    return input_sequence, output_sequence


def generate_copy_data_set(
        dimension,
        max_length_of_original_sequence,
        data_set_size):
    """Generate samples for learning copy algorithm.

    # Arguments
        dimension: the dimension of each input output tokens.
        max_length_of_original_sequence: the max length of original sequence.
        data_set_size: the size of total samples.

    # Returns
        input_sequences: the input sequences of total samples.
        output_sequences: the output sequences of total samples.
    """
    # get random sequence lengths from uniform distribution e.g. [1, 20]
    sequence_lengths = np.random.randint(
        1, max_length_of_original_sequence + 1, data_set_size)

    # allocate space for input sequences and output sequences, where the
    # "1" is a special flag which indicate the end of the input or output
    input_sequences = np.zeros(
        (data_set_size, max_length_of_original_sequence * 2 + 1, dimension),
        dtype=np.bool)
    output_sequences = np.zeros(
        (data_set_size, max_length_of_original_sequence * 2 + 1, dimension),
        dtype=np.bool)

    # set the value for input sequences and output sequences
    for i in range(data_set_size):
        input_sequence, output_sequence = \
            generate_copy_sample(dimension, sequence_lengths[i])
        input_sequences[i, :sequence_lengths[i]*2+1] = input_sequence
        output_sequences[i, :sequence_lengths[i]*2+1] = output_sequence

    # return the total samples
    return input_sequences, output_sequences


def generate_repeat_copy_sample(dimension, sequence_length, repeat_times):
    """Generate one sample of repeat copy algorithm.

    # Arguments
        dimension: the dimension of each input output tokens.
        sequence_length: the length of input sequence, i.e. the number of
                input tokens.
        repeat_times: repeat times of output.

    # Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    # produce random sequence
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, dimension - 1)).astype(np.uint8)

    # allocate space for input sequence and output sequence
    input_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * repeat_times,  # + 1
         dimension),
        dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * repeat_times,  # + 1
         dimension),
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

    # return the sample
    return input_sequence, output_sequence


def generate_repeat_copy_data_set(
        dimension,
        max_length_of_original_sequence,
        max_repeat_times,
        data_set_size):
    """Generate samples for learning repeat copy algorithm.

    # Arguments
        dimension: the dimension of each input output tokens.
        max_length_of_original_sequence: the max length of original sequence.
        max_repeat_times: the maximum repeat times.
        data_set_size: the size of total samples.

    # Returns
        input_sequences: the input sequences of total samples.
        output_sequences: the output sequences of total samples.
        repeat_times: the repeat times of each output sequence of total
            samples.
    """
    # produce random sequence lengths from uniform distribution
    # [1, max_length]
    sequence_lengths = np.random.randint(
        1, max_length_of_original_sequence + 1, data_set_size)

    # produce random repeat times from uniform distribution
    # [1, max_repeat_times]
    repeat_times = np.random.randint(1, max_repeat_times + 1, data_set_size)
    input_sequences = np.zeros(
        (data_set_size,
         max_length_of_original_sequence * (max_repeat_times + 1) + 1,  # + 1
         dimension),
        dtype=np.bool)
    output_sequences = np.zeros(
        (data_set_size,
         max_length_of_original_sequence * (max_repeat_times + 1) + 1,  # + 1
         dimension),
        dtype=np.bool)

    # set the value for input sequences and output sequences
    for i in range(data_set_size):
        input_sequence, output_sequence = generate_repeat_copy_sample(
            dimension, sequence_lengths[i], repeat_times[i])
        input_sequences[i, :sequence_lengths[i]*(repeat_times[i]+1)+1] = \
            input_sequence
        output_sequences[i, :sequence_lengths[i]*(repeat_times[i]+1)+1] = \
            output_sequence

    # return total samples
    return input_sequences, output_sequences, repeat_times


def generate_associative_recall_items(dimension, item_size, episode_size):
    """Generate items of associative recall algorithm.

    # Arguments
        dimension: the dimension of input output sequences.
        item_size: the size of items.
        episode_size: the size of one episode.

    # Returns
        items: the generated item.
    """
    inner_item = np.random.binomial(
        1, 0.5, ((item_size + 1) * episode_size, dimension)
    ).astype(np.uint8)
    items = np.zeros(((item_size + 1) * episode_size, dimension + 2),
                     dtype=np.uint8)
    # items = np.zeros(((item_size + 1) * episode_size, dimension + 2),
    #                  dtype=np.bool)
    items[:, :-2] = inner_item

    separator = np.zeros((1, dimension + 2), dtype=np.uint8)
    # separator = np.zeros((1, dimension + 2), dtype=np.bool)
    separator[0][-2] = 1
    items[:(item_size + 1) * episode_size:(item_size + 1)] = separator[0]

    # return one items for associative recall
    return items


def generate_associative_recall_sample(
        dimension, item_size, episode_size, max_episode_size):
    """Generate one sample of associative recall algorithm.

    Arguments
        dimension: the dimension of input output sequences.
        item_size: the size of one item.
        episode_size: the size of one episode.
        max_episode_size: the maximum episode size.

    Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    sequence_length = (item_size+1) * (max_episode_size+2)
    input_sequence = np.zeros(
        (sequence_length, dimension + 2), dtype=np.uint8)
    # input_sequence = np.zeros(
    #     (sequence_length, dimension + 2), dtype=np.bool)
    input_sequence[:(item_size + 1) * episode_size] = \
        generate_associative_recall_items(
            dimension, item_size, episode_size)

    separator = np.zeros((1, dimension + 2), dtype=np.uint8)
    # separator = np.zeros((1, dimension + 2), dtype=np.bool)
    separator[0][-2] = 1
    query_index = np.random.randint(0, episode_size-1)

    input_sequence[(item_size+1)*episode_size:(item_size+1)*(episode_size+1)] = \
        input_sequence[(item_size+1)*query_index:(item_size+1)*(query_index+1)]
    input_sequence[(item_size+1)*episode_size][-2] = 0
    input_sequence[(item_size+1)*episode_size][-1] = 1
    input_sequence[(item_size+1)*(episode_size+1)][-1] = 1

    output_sequence = np.zeros(
        (sequence_length, dimension + 2), dtype=np.uint8)
    # output_sequence = np.zeros(
    #     (sequence_length, dimension + 2), dtype=np.bool)
    output_sequence[(item_size+1)*(episode_size+1):(item_size+1)*(episode_size+2)] = \
        input_sequence[(item_size+1)*(query_index+1):(item_size+1)*(query_index+2)]
    output_sequence[(item_size+1)*(episode_size+1)][-2] = 0

    # return one sample for associative recall
    return input_sequence, output_sequence


def generate_associative_recall_data_set(
        dimension, item_size, max_episode_size, data_set_size):
    """Generate samples for learning associative recall algorithm.

    Arguments
        dimension: the dimension of input output sequences.
        item_size: the size of one item.
        max_episode_size: the maximum episode size.
        data_set_size: the size of one episode.

    Returns
        input_sequences: the input sequences of total samples.
        output_sequences: the output sequences of total samples.
    """
    episode_size = np.random.randint(2, max_episode_size + 1, data_set_size)
    sequence_length = (item_size+1) * (max_episode_size+2)
    input_sequences = np.zeros(
        (data_set_size, sequence_length, dimension + 2), dtype=np.uint8)
    output_sequences = np.zeros(
        (data_set_size, sequence_length, dimension + 2), dtype=np.uint8)
    # input_sequences = np.zeros(
    #   (training_size, sequence_length, dimension + 2), dtype=np.bool)
    # output_sequences = np.zeros(
    #   (training_size, sequence_length, dimension + 2), dtype=np.bool)
    for i in range(data_set_size):
        input_sequence, output_sequence = generate_associative_recall_sample(
            dimension, item_size, episode_size[i], max_episode_size)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence

    # return the total samples
    return input_sequences, output_sequences


def generate_priority_sort_sample(
        dimension,
        input_sequence_length,
        output_sequence_length,
        priority_lower_bound,
        priority_upper_bound):
    """Generate one sample of priority sort algorithm.

    Arguments
        dimension: the dimension of input output sequences.
        input_sequence_length: the length of input sequence.
        output_sequence_length: the length of output sequence.
        priority_lower_bound: the lower bound of priority.
        priority_upper_bound: the upper bound of priority.

    Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    sequence = input_sequence_length + output_sequence_length + 1
    input_sequence = np.random.binomial(
        1, 0.5, (input_sequence_length, dimension + 1)).astype(np.uint8)
    output_sequence = np.zeros(
        (output_sequence_length, dimension + 1), dtype=np.uint8)
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

    input_sequence_ = np.zeros((sequence, dimension + 2), dtype=np.float32)
    input_priority_ = np.zeros((sequence, 1), dtype=np.float32)
    output_sequence_ = np.zeros((sequence, dimension + 2), dtype=np.float32)
    output_priority_ = np.zeros((sequence, 1), dtype=np.float32)

    input_sequence_[:input_sequence_length, :-1] = input_sequence
    input_sequence_[input_sequence_length][-1] = 1
    input_priority_[:input_sequence_length] = input_priority
    output_sequence_[input_sequence_length+1:sequence, :-1] = output_sequence
    output_priority_[input_sequence_length+1:sequence] = output_priority

    # return input sequence, priority of each input, output sequence, priority
    # of each output
    return input_sequence_, input_priority_, output_sequence_, output_priority_


def generate_priority_sort_data_set(
        dimension,
        input_sequence_length,
        output_sequence_length,
        priority_lower_bound,
        priority_upper_bound,
        data_set_size):
    """Generate samples for learning priority sort algorithm.

    Arguments
        dimension: the dimension of input output sequences.
        input_sequence_length: the length of input sequence.
        output_sequence_length: the length of output sequence.
        priority_lower_bound: the lower bound of priority.
        priority_upper_bound: the upper bound of priority.
        data_set_size: the size of one episode.

    Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    sequence_length = input_sequence_length + output_sequence_length
    input_sequences = np.zeros(
        (data_set_size, sequence_length + 1, dimension + 2), dtype=np.float32)
    output_sequences = np.zeros(
        (data_set_size, sequence_length + 1, dimension + 2), dtype=np.float32)
    for i in range(data_set_size):
        input_sequence, input_priority, output_sequence, output_priority = \
            generate_priority_sort_sample(
                dimension,
                input_sequence_length,
                output_sequence_length,
                priority_lower_bound,
                priority_upper_bound)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence
        input_sequences[i][:, -2] = input_priority.transpose()
        output_sequences[i][:, -2] = output_priority.transpose()

    # return the total samples
    return input_sequences, output_sequences