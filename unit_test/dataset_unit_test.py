import dataset
import visualization
import numpy as np


def test_copy_data_generation():
    input_sequence, output_sequence = dataset.generate_copy_data(8, 10)
    print input_sequence
    print output_sequence
    input_sequence, output_sequence = dataset.generate_copy_data(8, 20)
    print input_sequence
    print output_sequence
    input_sequence, output_sequence = dataset.generate_copy_data(8, 30)
    print input_sequence
    print output_sequence
    input_sequence, output_sequence = dataset.generate_copy_data(8, 50)
    print input_sequence
    print output_sequence
    input_sequence, output_sequence = dataset.generate_copy_data(8, 120)
    print input_sequence
    print output_sequence


def show_repeat_copy_data_generation():
    print('Generating data...')
    input_sequence, output_sequence, repeat_times = \
        dataset.generate_repeat_copy_data_set(7, 10, 100, 10)
    print(output_sequence[0].transpose().shape)

    for i in range(100):
        visualization.show_repeat_copy_data(
            output_sequence[i].transpose(),
            input_sequence[i].transpose(),
            "Output sequence $y^{(t)}$",
            "Input sequence $x^{(t)}$",
            "../experiment/repeat_copy_data_predict_%2d.pdf"%i,
            repeat_times[i]
        )


def test_repeat_copy_data_generation():
    print('Generating data...')
    input_sequence, output_sequence, repeat_times = \
        dataset.generate_repeat_copy_data_set(4, 10, 20, 20)

    print(input_sequence.shape)
    matrix_list = []
    matrix_list.append(input_sequence[0].transpose())
    matrix_list.append(output_sequence[0].transpose())
    matrix_list.append(output_sequence[0].transpose())
    name_list = []
    name_list.append("Input")
    name_list.append("Target")
    name_list.append("Predict")
    show_matrix = visualization.PlotDynamicalMatrix4Repeat(matrix_list, name_list, repeat_times[0])

    for i in range(20):
        matrix_list_update = []
        matrix_list_update.append(input_sequence[i].transpose())
        matrix_list_update.append(output_sequence[i].transpose())
        matrix_list_update.append(output_sequence[i].transpose())
        show_matrix.update(matrix_list_update, name_list, repeat_times[i])
        show_matrix.save("../experiment/repeat_copy_data_predict_%2d.png"%i)


def test_associative_recall_data():
    INPUT_DIMENSION_SIZE = 6
    ITEM_SIZE = 4
    MAX_EPISODE_SIZE = 20
    TRAINING_SIZE = 100
    MAX_INPUT_LENGTH = (ITEM_SIZE+1) * (MAX_EPISODE_SIZE+2)
    print(MAX_INPUT_LENGTH)
    # item = dataset.generate_associative_recall_items(input_size, item_size, episode_size)
    # print(item)

    print('Generating data sets...')
    input_sequence, output_sequence = dataset.generate_associative_recall_data_set(
        INPUT_DIMENSION_SIZE, ITEM_SIZE, MAX_EPISODE_SIZE, TRAINING_SIZE)
    # print input_sequence
    # print output_sequence

    print(input_sequence.shape)
    print(output_sequence.shape)

    for i in range(TRAINING_SIZE):
        # print(output_sequence[i].transpose())
        # print(input_sequence[i].transpose())
        visualization.show_associative_recall_data(
            output_sequence[i].transpose(),
            input_sequence[i].transpose(),
            "Output sequence $y^{(t)}$",
            "Input sequence $x^{(t)}$",
            "../experiment/associative_recall_%2d.pdf"%i
        )


def test_n_gram_data():
    a = 0.5
    b = 0.5
    n = 6
    look_up_table = dataset.generate_probability_of_n_gram_by_beta(a, b, n)
    sequence_length = 50
    example_size = 100
    # print(look_up_table)
    train_X, train_Y = dataset.generate_dynamical_n_gram_data_set(
        look_up_table, n, sequence_length, example_size)
    # print(train_X)
    show_matrix = visualization.PlotDynamicalMatrix4NGram(
        train_X[0].transpose(), train_Y[0].transpose(), train_Y[0].transpose())

    for i in range(example_size):
        show_matrix.update(train_X[i].transpose(), train_Y[i].transpose(), train_Y[i].transpose())
        show_matrix.save("../experiment/n_gram_data_training_%2d.png"%i)

    show_matrix.close()


def test_priority_sort_data():
    input_size = 8
    input_sequence_length = 20
    output_sequence_length = 16
    priority_lower_bound = -1
    priority_upper_bound = 1
    example_size = 10
    input_matrix = np.zeros((input_sequence_length+1, input_size+2), dtype=np.float32)
    output_matrix = np.zeros((output_sequence_length+1, input_size+2), dtype=np.float32)

    train_x_seq, train_y_seq = \
        dataset.generate_associative_priority_sort_data_set(
            input_size,
            input_sequence_length,
            output_sequence_length,
            priority_lower_bound,
            priority_upper_bound,
            example_size)

    print(train_x_seq[0].shape)
    print(input_matrix.shape)
    input_matrix = train_x_seq[0]
    output_matrix = train_y_seq[0]
    show_matrix = visualization.PlotDynamicalMatrix4PrioritySort(
        input_matrix.transpose(),
        output_matrix.transpose(),
        output_matrix.transpose())
    for i in range(example_size):
        input_matrix = train_x_seq[i]
        output_matrix = train_y_seq[i]
        # input_matrix[:, :-1] = train_x_seq[i]
        # input_matrix[:, -1] = train_x_priority[i].reshape(input_sequence_length)
        # output_matrix[:, :-1] = train_y_seq[i]
        # output_matrix[:, -1] = train_y_priority[i].reshape(output_sequence_length)
        show_matrix.update(input_matrix.transpose(),
                           output_matrix.transpose(),
                           output_matrix.transpose())
        show_matrix.save("../experiment/priority_data_training_%2d.png"%i)

    show_matrix.close()


if __name__ == "__main__":
    # test_copy_data_generation()
    show_repeat_copy_data_generation()
    # test_repeat_copy_data_generation()
    # test_associative_recall_data()
    # test_n_gram_data()
    # test_priority_sort_data()
