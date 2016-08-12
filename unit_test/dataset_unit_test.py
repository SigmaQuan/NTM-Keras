import dataset


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
    input_sequence, output_sequence = dataset.generate_repeat_copy_data(8, 16, 2)
    print input_sequence
    print output_sequence


if __name__ == "__main__":
    test_copy_data_generation()
    test_repeat_copy_data_generation()
