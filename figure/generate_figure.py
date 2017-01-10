import dataset
import visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def show_repeat_copy_data(
        output_sequence_copy,
        input_sequence_copy,
        repeat_times_copy,
        output_sequence_recall,
        input_sequence_recall,
        output_sequence_sort,
        input_sequence_sort,
        input_name,
        output_name,
        image_file):
    # set figure size
    fig = plt.figure(figsize=(12, 5))
    #
    # # draw first line
    # axes_input_10 = plt.subplot2grid((9, 1), (1, 0), colspan=1)
    # axes_input_10.set_aspect('equal')
    # plt.imshow(output_sequence_copy, interpolation='none')
    # axes_input_10.set_xticks([])
    # axes_input_10.set_yticks([])
    # # draw second line
    # axes_output_10 = plt.subplot2grid((9, 1), (2, 0), colspan=1)
    # plt.imshow(input_sequence_copy, interpolation='none')
    # axes_output_10.set_xticks([])
    # axes_output_10.set_yticks([])
    # # draw third line
    # # plt.text(80, 12, "(a) repeat copy", ha='center')
    # # title = "Repeat times = %d" % repeat_times_copy
    # # plt.text(80, -12, title, ha='center')
    # # plt.text(-2, 5, output_name, ha='right')
    # # plt.text(-2, -5, input_name, ha='right')
    # # # plt.text(18, 12, 'Time $t$ $\longrightarrow$', ha='right')
    # # plt.text(9, 12, '$t$ $\longrightarrow$', ha='right')
    #
    # # draw first line
    # axes_input_10 = plt.subplot2grid((9, 1), (4, 0), colspan=1)
    # axes_input_10.set_aspect('equal')
    # plt.imshow(output_sequence_recall, interpolation='none')
    # axes_input_10.set_xticks([])
    # axes_input_10.set_yticks([])
    # # draw second line
    # axes_output_10 = plt.subplot2grid((9, 1), (5, 0), colspan=1)
    # plt.imshow(input_sequence_recall, interpolation='none')
    # axes_output_10.set_xticks([])
    # axes_output_10.set_yticks([])
    # # draw third line
    # # plt.text(80, 12, "(b) associative recall", ha='center')
    # # plt.text(-2, 5, output_name, ha='right')
    # # plt.text(-2, -5, input_name, ha='right')
    # # plt.text(9, 12, '$t$ $\longrightarrow$', ha='right')

    # draw first line
    axes_input_10 = plt.subplot2grid((9, 1), (7, 0), colspan=1)
    axes_input_10.set_aspect('equal')
    plt.imshow(output_sequence_sort, interpolation='none')
    axes_input_10.set_xticks([])
    axes_input_10.set_yticks([])
    # draw second line
    axes_output_10 = plt.subplot2grid((9, 1), (8, 0), colspan=1)
    plt.imshow(input_sequence_sort, interpolation='none')
    axes_output_10.set_xticks([])
    axes_output_10.set_yticks([])
    # draw third line
    # plt.text(80, 12, "(c) priority sort", ha='center')
    # plt.text(-2, 5, output_name, ha='right')
    # plt.text(-2, -5, input_name, ha='right')
    # plt.text(9, 12, '$t$ $\longrightarrow$', ha='right')

    # add color bars
    # # *rect* = [left, bottom, width, height]
    # cax = plt.axes([0.84, 0.1, 0.005, 0.71])
    cax = plt.axes([0.84, 0.1, 0.005, 0.165])
    cbar = plt.colorbar(cax=cax)
    # show colorbar
    # cbar = plt.colorbar(gci)
    # cbar.set_label('$T_B(K)$', fontdict=font)
    cbar.set_ticks(np.linspace(0, 1, 3))
    cbar.set_ticklabels(('0', '0.5', '1'))

    # show figure
    plt.show()

    # save image
    fig.savefig(image_file, dpi=75, format='pdf')

    # close plot GUI
    plt.close()


def show_algorithm_learning_example():
    input_size_copy = 8
    sequence_length_copy = 10
    repeat_times = 15
    input_sequence_copy, output_sequence_copy = \
        dataset.generate_repeat_copy_data(
            input_size_copy, sequence_length_copy, repeat_times)
    print(input_sequence_copy.shape)
    print(output_sequence_copy.shape)

    input_size_recall = 6
    # item_size = 4
    item_size = 3
    episode_size = 38
    max_episode_size = 38
    input_sequence_recall = np.zeros(input_sequence_copy.shape)
    output_sequence_recall = np.zeros(output_sequence_copy.shape)
    input_sequence_recall_, output_sequence_recall_ = \
        dataset.generate_associative_recall_data(
            input_size_recall, item_size, episode_size, max_episode_size)
    input_sequence_recall[:-1] = input_sequence_recall_
    output_sequence_recall[:-1] = output_sequence_recall_
    print(input_sequence_recall.shape)
    print(output_sequence_recall.shape)

    input_size_sort = 6
    input_sequence_length = 80
    output_sequence_length = 80
    priority_lower_bound = 0
    priority_upper_bound = 1
    # input_sequence_sort = np.zeros(input_sequence_copy.shape)
    # output_sequence_sort = np.zeros(output_sequence_copy.shape)
    input_sequence_sort_, input_priority_, output_sequence_sort_, output_priority_ = \
        dataset.generate_associative_priority_sort_data(
            input_size_sort,
            input_sequence_length,
            output_sequence_length,
            priority_lower_bound,
            priority_upper_bound)

    sequence_length = input_sequence_length + output_sequence_length
    input_sequence_sort = np.zeros((sequence_length+1, input_size_sort+2), dtype=np.float32)
    output_sequence_sort = np.zeros((sequence_length+1, input_size_sort+2), dtype=np.float32)
    input_sequence_sort = input_sequence_sort_
    output_sequence_sort = output_sequence_sort_
    input_sequence_sort[:, -2] = input_priority_.transpose()[0]
    output_sequence_sort[:, -2] = output_priority_.transpose()[0]
    print(input_sequence_sort.shape)
    print(output_sequence_sort.shape)

    # print(input_sequence_sort[1:50, :])
    print(input_sequence_sort[:, -2])
    print(input_priority_.transpose()[0])
    show_repeat_copy_data(
        output_sequence_copy.transpose(),
        input_sequence_copy.transpose(),
        repeat_times,
        output_sequence_recall.transpose(),
        input_sequence_recall.transpose(),
        output_sequence_sort.transpose(),
        input_sequence_sort.transpose(),
        "$y^{(t)}$",
        "$x^{(t)}$",
        "../experiment/algorithm_learning_data.pdf")
    print("end..")

    # file_priority_input_sequence = "../experiment/file_priority_input_sequence.txt"
    # file_priority_output_sequence = "../experiment/file_priority_output_sequence.txt"
    #
    # priority_input_sequence = open(file_priority_input_sequence, 'w')
    # (row, column) = input_sequence_sort.shape
    # for i in range(row):
    #     for j in range(column):
    #         one_point = "%d %d %f\n"%(i, j, input_sequence_sort[i][j])
    #         priority_input_sequence.write(one_point)
    # priority_input_sequence.close()


if __name__ == "__main__":
    show_algorithm_learning_example()
