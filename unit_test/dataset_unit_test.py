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
    input_sequence, output_sequence = dataset.generate_repeat_copy_data(8, 16, 2)
    print(input_sequence.shape)
    # print(input_sequence)
    # print(output_sequence)
    matrix_list = []
    matrix_list.append(input_sequence.transpose())
    matrix_list.append(output_sequence.transpose())
    matrix_list.append(output_sequence.transpose())
    name_list = []
    name_list.append("Input")
    name_list.append("Target")
    name_list.append("Predict")
    show_matrix = visualization.PlotDynamicalMatrix(matrix_list, name_list)
    show_matrix.update(matrix_list, name_list)
    show_matrix.update(matrix_list, name_list)
    show_matrix.update(matrix_list, name_list)
    show_matrix.update(matrix_list, name_list)
    show_matrix.save("../experiment/repeat_copy_data_sample.png")


if __name__ == "__main__":
    # test_copy_data_generation()
    test_repeat_copy_data_generation()
