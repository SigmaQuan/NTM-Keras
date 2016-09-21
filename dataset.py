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
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.bool)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = 1
    output_sequence[sequence_length + 1:, :-1] = sequence
    # output_sequence[:sequence_length + 1, -1] = 1
    output_sequence[sequence_length, -1] = 1

    return input_sequence, output_sequence


def generate_copy_data_set(input_size, max_size, training_size):
    sequence_lengths = np.random.randint(1, max_size + 1, training_size)
    input_sequences = np.zeros((training_size, max_size*2+1, input_size), dtype=np.bool)
    output_sequences = np.zeros((training_size, max_size*2+1, input_size), dtype=np.bool)
    for i in range(training_size):
        input_sequence, output_sequence = generate_copy_data(
            input_size, sequence_lengths[i])
        for j in range(sequence_lengths[i]*2+1):
            # index_1 = max_size*2+1-j-1
            # index_2 = sequence_lengths[i]*2+1-j-1
            index_1 = j
            index_2 = j
            input_sequences[i][index_1] = input_sequence[index_2]
            output_sequences[i][index_1] = output_sequence[index_2]
        # for k in range(sequence_lengths[i]*2+1, max_size*2+1):
        #     input_sequences[i][k][-1] = 1
        #     output_sequences[i][k][-1] = 1
        # for k in range(sequence_lengths[i]*2+1, max_size*2+1):
        input_sequences[i][sequence_lengths[i]*2+1][-1] = 1
        output_sequences[i][sequence_lengths[i]*2+1][-1] = 1

    return input_sequences, output_sequences


def generate_repeat_copy_data(input_size, sequence_length, repeat_times):
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * repeat_times + 1, input_size), dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * repeat_times + 1, input_size), dtype=np.bool)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = repeat_times
    output_sequence[sequence_length + 1:-1, :-1] = \
        np.tile(sequence, (repeat_times, 1))
    output_sequence[-1, -1] = 1

    return input_sequence, output_sequence


# # Fixed repeat size
# def generate_repeat_copy_data_set(input_size, max_size, training_size, repeat_times):
#     sequence_lengths = np.random.randint(1, max_size + 1, training_size)
#     input_sequences = np.zeros((training_size, max_size*(repeat_times+1)+1+1, input_size), dtype=np.bool)
#     output_sequences = np.zeros((training_size, max_size*(repeat_times+1)+1+1, input_size), dtype=np.bool)
#     for i in range(training_size):
#         input_sequence, output_sequence = generate_repeat_copy_data(
#             input_size, sequence_lengths[i], repeat_times)
#         for j in range(sequence_lengths[i]*(repeat_times+1)+1):
#             index_1 = j
#             index_2 = j
#             input_sequences[i][index_1] = input_sequence[index_2]
#             output_sequences[i][index_1] = output_sequence[index_2]
#         # input_sequences[i][sequence_lengths[i]*(repeat_times+1)+1][-1] = 1
#         output_sequences[i][sequence_lengths[i]*(repeat_times+1)+1][-1] = 1
#     return input_sequences, output_sequences


def generate_repeat_copy_data_set(input_size, max_size, training_size, max_repeat_times):
    sequence_lengths = np.random.randint(1, max_size + 1, training_size)
    repeat_times = np.random.randint(1, max_repeat_times + 1, training_size)
    input_sequences = np.zeros((training_size, max_size*(max_repeat_times+1)+1+1, input_size), dtype=np.bool)
    output_sequences = np.zeros((training_size, max_size*(max_repeat_times+1)+1+1, input_size), dtype=np.bool)
    for i in range(training_size):
        input_sequence, output_sequence = generate_repeat_copy_data(
            input_size, sequence_lengths[i], repeat_times[i])
        for j in range(sequence_lengths[i]*(repeat_times[i]+1)+1):
            index_1 = j
            index_2 = j
            input_sequences[i][index_1] = input_sequence[index_2]
            output_sequences[i][index_1] = output_sequence[index_2]
        # input_sequences[i][sequence_lengths[i]*(repeat_times+1)+1][-1] = 1
        output_sequences[i][sequence_lengths[i]*(repeat_times[i]+1)+1][-1] = 1
    return input_sequences, output_sequences, repeat_times


def generate_associative_recall_items(input_size, item_size, episode_size):
    inner_item = np.random.binomial(1, 0.5,
                                    ((item_size + 1) * episode_size, input_size)
                                    ).astype(np.uint8)
    items = np.zeros(((item_size + 1) * episode_size, input_size + 2), dtype=np.uint8)
    # item = np.zeros(((item_size + 1) * episode_size, input_size + 2), dtype=np.bool)
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
        generate_associative_recall_items(input_size, item_size, episode_size)

    separator = np.zeros((1, input_size+2), dtype=np.uint8)
    # separator = np.zeros((1, input_size + 2), dtype=np.bool)
    separator[0][-2] = 1
    query_index = np.random.randint(0, episode_size-1)

    input_sequence[(item_size+1)*episode_size:(item_size+1)*(episode_size+1)] = \
        input_sequence[(item_size+1)*query_index:(item_size+1)*(query_index+1)]
    input_sequence[(item_size+1)*(episode_size)][-2] = 0
    input_sequence[(item_size+1)*(episode_size)][-1] = 1
    input_sequence[(item_size+1)*(episode_size+1)][-1] = 1

    output_sequence = np.zeros(
        (sequence_length, input_size + 2), dtype=np.uint8)
    # output_sequence = np.zeros(
    #     (sequence_length, input_size + 2), dtype=np.bool)
    output_sequence[(item_size+1)*(episode_size+1):(item_size+1)*(episode_size+2)] = \
        input_sequence[(item_size+1)*(query_index+1):(item_size+1)*(query_index+2)]
    output_sequence[(item_size+1)*(episode_size+1)][-2] = 0

    return input_sequence, output_sequence


def generate_repeat_copy_data_set(
        input_size, item_size, max_episode_size, training_size):
    episode_size = np.random.randint(2, max_episode_size + 1, training_size)
    sequence_length = (item_size+1) * (max_episode_size+2)
    input_sequences = np.zeros((training_size, sequence_length, input_size + 2), dtype=np.uint8)
    output_sequences = np.zeros((training_size, sequence_length, input_size + 2), dtype=np.uint8)
    # input_sequences = np.zeros((training_size, sequence_length, input_size + 2), dtype=np.bool)
    # output_sequences = np.zeros((training_size, sequence_length, input_size + 2), dtype=np.bool)
    for i in range(training_size):
        input_sequence, output_sequence = generate_associative_recall_data(
            input_size, item_size, episode_size[i], max_episode_size)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence

    return input_sequences, output_sequences