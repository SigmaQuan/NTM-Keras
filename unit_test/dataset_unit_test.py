import dataset
import visualization

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


def test_repeat_copy_data_generation():
    print('Generating data...')
    input_sequence, output_sequence, repeat_times = \
        dataset.generate_repeat_copy_data_set(4, 10, 20, 5)

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


if __name__ == "__main__":
    # test_copy_data_generation()
    test_repeat_copy_data_generation()
