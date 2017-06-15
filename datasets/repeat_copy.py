# -*- coding: utf-8 -*-
import numpy as np
from utils import initialize_random_seed


# Initialize the random seed
initialize_random_seed()


def generate_one_sample(dimension, sequence_length, repeat_times):
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


def generate_data_set(
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
        input_sequence, output_sequence = generate_one_sample(
            dimension, sequence_lengths[i], repeat_times[i])
        input_sequences[i, :sequence_lengths[i]*(repeat_times[i]+1)+1] = \
            input_sequence
        output_sequences[i, :sequence_lengths[i]*(repeat_times[i]+1)+1] = \
            output_sequence

    # return total samples
    return input_sequences, output_sequences, repeat_times
