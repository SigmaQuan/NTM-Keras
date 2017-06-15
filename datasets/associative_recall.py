# -*- coding: utf-8 -*-
import numpy as np
from utils import initialize_random_seed


# Initialize the random seed
initialize_random_seed()


def generate_items(dimension, item_size, episode_size):
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


def generate_one_sample(
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
        generate_items(
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


def generate_data_set(
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
        input_sequence, output_sequence = generate_one_sample(
            dimension, item_size, episode_size[i], max_episode_size)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence

    # return the total samples
    return input_sequences, output_sequences