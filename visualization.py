"""
Visualization of Neural Turing Machines.
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def show(w, w_title):
    """
    Show a weight matrix.
    :param w: the weight matrix.
    :param w_title: the title of the weight matrix
    :return: None.
    """
    # show w_z matrix of update gate.
    axes_w = plt.gca()
    plt.imshow(w)
    plt.colorbar()
    # plt.colorbar(orientation="horizontal")
    plt.xlabel("$w_{1}$")
    plt.ylabel("$w_{2}$")
    axes_w.set_xticks([])
    axes_w.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w[0]), len(w))
    w_title += matrix_size
    plt.title(w_title)

    # show the matrix.
    plt.show()


def make_tick_labels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


def show_copy_data(input_sequence, output_sequence, input_name, output_name, image_file):
    # set figure size
    fig = plt.figure(figsize=(7, 3))

    # draw first line
    axes_input_10 = plt.subplot2grid((2, 1), (0, 0), colspan=1)
    axes_input_10.set_aspect('equal')
    plt.imshow(input_sequence, interpolation='none')
    axes_input_10.set_xticks([])
    axes_input_10.set_yticks([])

    # draw second line
    axes_output_10 = plt.subplot2grid((2, 1), (1, 0), colspan=1)
    plt.imshow(output_sequence, interpolation='none')
    axes_output_10.set_xticks([])
    axes_output_10.set_yticks([])

    # add text
    plt.text(-2, -4.5, input_name, ha='right')
    plt.text(-2, 4, output_name, ha='right')
    plt.text(6, 10, 'Time $\longrightarrow$', ha='right')

    # set tick labels invisible
    make_tick_labels_invisible(plt.gcf())
    # adjust spaces
    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1, right=0.8, top=0.9)
    # add color bars
    # *rect* = [left, bottom, width, height]
    cax = plt.axes([0.85, 0.125, 0.015, 0.75])
    plt.colorbar(cax=cax)

    # show figure
    # plt.show()

    # save image
    # pp = PdfPages(image_file)
    # plt.savefig(pp, format='pdf')
    # pp.close()
    fig.savefig(image_file, dpi=75)

    # close plot GUI
    plt.close()


def show_multi_copy_data(target_sequence_10, output_sequence_10,
                   target_sequence_20, output_sequence_20,
                   target_sequence_30, output_sequence_30,
                   target_sequence_50, output_sequence_50,
                   target_sequence_120, output_sequence_120,
                   image_file):
    # set figure size
    fig = plt.figure(figsize=(12, 4))

    # draw first line
    axes_target_10 = plt.subplot2grid((4, 11), (0, 0), colspan=1)
    axes_target_10.set_aspect('equal')
    plt.imshow(target_sequence_10, interpolation='none')
    axes_target_10.set_xticks([])
    axes_target_10.set_yticks([])
    axes_target_20 = plt.subplot2grid((4, 11), (0, 1), colspan=2)
    plt.imshow(target_sequence_20, interpolation='none')
    axes_target_20.set_xticks([])
    axes_target_20.set_yticks([])
    axes_target_30 = plt.subplot2grid((4, 11), (0, 3), colspan=3)
    plt.imshow(target_sequence_30, interpolation='none')
    axes_target_30.set_xticks([])
    axes_target_30.set_yticks([])
    axes_target_50 = plt.subplot2grid((4, 11), (0, 6), colspan=5)
    plt.imshow(target_sequence_50, interpolation='none')
    axes_target_50.set_xticks([])
    axes_target_50.set_yticks([])

    # draw second line
    axes_output_10 = plt.subplot2grid((4, 11), (1, 0), colspan=1)
    plt.imshow(output_sequence_10, interpolation='none')
    axes_output_10.set_xticks([])
    axes_output_10.set_yticks([])
    axes_output_20 = plt.subplot2grid((4, 11), (1, 1), colspan=2)
    plt.imshow(output_sequence_20, interpolation='none')
    axes_output_20.set_xticks([])
    axes_output_20.set_yticks([])
    axes_output_30 = plt.subplot2grid((4, 11), (1, 3), colspan=3)
    plt.imshow(output_sequence_30, interpolation='none')
    axes_output_30.set_xticks([])
    axes_output_30.set_yticks([])
    axes_output_50 = plt.subplot2grid((4, 11), (1, 6), colspan=5)
    plt.imshow(output_sequence_50, interpolation='none')
    axes_output_50.set_xticks([])
    axes_output_50.set_yticks([])

    # draw last two lines
    axes_target_120 = plt.subplot2grid((4, 11), (2, 0), colspan=11)
    plt.imshow(target_sequence_120, interpolation='none')
    axes_target_120.set_xticks([])
    axes_target_120.set_yticks([])
    axes_output_120 = plt.subplot2grid((4, 11), (3, 0), colspan=11)
    plt.imshow(output_sequence_120, interpolation='none')
    axes_output_120.set_xticks([])
    axes_output_120.set_yticks([])

    # add text
    plt.text(-2, 5, 'Outputs', ha='right')
    plt.text(-2, -7.5, 'Targets', ha='right')
    plt.text(-2, -20, 'Outputs', ha='right')
    plt.text(-2, -32.5, 'Targets', ha='right')
    plt.text(10, 12, 'Time $\longrightarrow$', ha='right')

    # set tick labels invisible
    make_tick_labels_invisible(plt.gcf())
    # adjust spaces
    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1, right=0.8, top=0.9)
    # add color bars
    # *rect* = [left, bottom, width, height]
    cax = plt.axes([0.85, 0.125, 0.015, 0.75])
    plt.colorbar(cax=cax)

    # show figure
    plt.show()

    # save image
    # pp = PdfPages(image_file)
    # plt.savefig(pp, format='pdf')
    # pp.close()
    fig.savefig(image_file, dpi=75)

    # close plot GUI
    plt.close()


def show_memory_of_copy_task(
        input_sequence, output_squence,
        adds, reads,
        write_weightings, read_weightings,
        image_file):
    # set figure size
    fig = plt.figure(figsize=(10, 8))

    # draw first line
    axes_input = plt.subplot2grid((15, 2), (0, 0), rowspan=2)
    plt.imshow(input_sequence, interpolation='none')
    axes_input.set_xticks([])
    axes_input.set_yticks([])
    plt.title("Inputs")
    axes_output = plt.subplot2grid((15, 2), (0, 1), rowspan=2)
    plt.imshow(output_squence, interpolation='none')
    axes_output.set_xticks([])
    axes_output.set_yticks([])
    plt.title("Outputs")

    # draw second line
    axes_adds = plt.subplot2grid((15, 2), (2, 0), rowspan=4)
    plt.imshow(adds)  # , interpolation='none'
    axes_adds.set_xticks([])
    axes_adds.set_yticks([])
    plt.title("Adds")
    axes_reads = plt.subplot2grid((15, 2), (2, 1), rowspan=4)
    plt.imshow(reads)  # , interpolation='none'
    axes_reads.set_xticks([])
    axes_reads.set_yticks([])
    plt.title("Reads")

    # draw last line
    axes_write = plt.subplot2grid((15, 2), (6, 0), rowspan=9)
    plt.imshow(write_weightings, interpolation='none')
    axes_write.set_xticks([])
    axes_write.set_yticks([])
    plt.title("Write Weightings")
    axes_read = plt.subplot2grid((15, 2), (6, 1), rowspan=9)
    plt.imshow(read_weightings, interpolation='none')
    axes_read.set_xticks([])
    axes_read.set_yticks([])
    plt.title("Read Weightings")

    # add text
    plt.text(-45, 20.5, 'Location $\longrightarrow$', fontsize=16, ha='center', rotation=90)
    plt.text(11.5, 39, 'Time $\longrightarrow$', fontsize=16, ha='right')
    plt.text(-30.5, 39, 'Time $\longrightarrow$', fontsize=16, ha='right')

    # set tick labels invisible
    make_tick_labels_invisible(plt.gcf())
    # adjust spaces
    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1, right=0.8, top=0.9)
    # add color bars
    # *rect* = [left, bottom, width, height]
    cax = plt.axes([0.85, 0.15, 0.015, 0.75])
    plt.colorbar(cax=cax)

    # show figure
    plt.show()

    # save image
    fig.savefig(image_file, dpi=75)

    # close plot GUI
    plt.close()


class PlotDynamicalMatrix:
    def __init__(self, matrix_list, name_list):
        """
        Initialize the value of matrix.
        :param matrix_list: a goup of matrix.
        :return: non.
        """
        self.matrix_list = matrix_list
        # set figure size
        self.fig = plt.figure(figsize=(7, 5))

        plt.ion()
        self.update(matrix_list, name_list)

    def update(self, matrix_list, name_list):
        # draw first line
        axes_input = plt.subplot2grid((3, 1), (0, 0), colspan=1)
        axes_input.set_aspect('equal')
        plt.imshow(matrix_list[0], interpolation='none')
        axes_input.set_xticks([])
        axes_input.set_yticks([])

        # draw second line
        axes_output = plt.subplot2grid((3, 1), (1, 0), colspan=1)
        plt.imshow(matrix_list[1], interpolation='none')
        axes_output.set_xticks([])
        axes_output.set_yticks([])

        # draw third line
        axes_predict = plt.subplot2grid((3, 1), (2, 0), colspan=1)
        plt.imshow(matrix_list[2], interpolation='none')
        axes_predict.set_xticks([])
        axes_predict.set_yticks([])

        # # add text
        # plt.text(-2, -19.5, name_list[0], ha='right')
        # plt.text(-2, -7.5, name_list[1], ha='right')
        # plt.text(-2, 4.5, name_list[2], ha='right')
        # plt.text(6, 10, 'Time $\longrightarrow$', ha='right')

        # set tick labels invisible
        make_tick_labels_invisible(plt.gcf())
        # adjust spaces
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1, right=0.8, top=0.9)
        # add color bars
        # *rect* = [left, bottom, width, height]
        cax = plt.axes([0.85, 0.125, 0.015, 0.75])
        plt.colorbar(cax=cax)

        # show figure
        # plt.show()
        plt.draw()
        plt.pause(0.025)
        # plt.pause(15)

    def save(self, image_file):
        # save image
        # pp = PdfPages(image_file)
        # plt.savefig(pp, format='pdf')
        # pp.close()
        self.fig.savefig(image_file, dpi=75)

    def close(self):
        # close plot GUI
        plt.close()


class PlotDynamicalMatrix4Repeat:
    def __init__(self, matrix_list, name_list, repeat_times):
        """
        Initialize the value of matrix.
        :param matrix_list: a goup of matrix.
        :return: non.
        """
        self.matrix_list = matrix_list
        # set figure size
        self.fig = plt.figure(figsize=(11, 3))

        plt.ion()
        self.update(matrix_list, name_list, repeat_times)

    def update(self, matrix_list, name_list, repeat_times):
        # draw first line
        axes_input = plt.subplot2grid((3, 1), (0, 0), colspan=1)
        axes_input.set_aspect('equal')
        plt.imshow(matrix_list[0], interpolation='none')
        axes_input.set_xticks([])
        axes_input.set_yticks([])

        # draw second line
        axes_output = plt.subplot2grid((3, 1), (1, 0), colspan=1)
        plt.imshow(matrix_list[1], interpolation='none')
        axes_output.set_xticks([])
        axes_output.set_yticks([])

        # draw third line
        axes_predict = plt.subplot2grid((3, 1), (2, 0), colspan=1)
        plt.imshow(matrix_list[2], interpolation='none')
        axes_predict.set_xticks([])
        axes_predict.set_yticks([])
        # for 8bits 20length
        # # add text
        # plt.text(-2, -22, name_list[0], ha='right')
        # plt.text(-2, -9, name_list[1], ha='right')
        # plt.text(-2, 4.5, name_list[2], ha='right')
        # plt.text(12, 12, 'Time $\longrightarrow$', ha='right')
        #
        # title = "Repeat Times = %d"%repeat_times
        # plt.text(60, -30, title, ha='center')
        # # plt.title(title)
        #

        # add text
        plt.text(-2, -11.3, name_list[0], ha='right')
        plt.text(-2, -4.8, name_list[1], ha='right')
        plt.text(-2, 2, name_list[2], ha='right')
        plt.text(5.5, 6, 'Time $\longrightarrow$', ha='right')

        title = "Repeat Times = %d"%repeat_times
        plt.text(30, -15, title, ha='center')
        # plt.title(title)

        # set tick labels invisible
        make_tick_labels_invisible(plt.gcf())
        # adjust spaces
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1, right=0.8, top=0.9)
        # add color bars
        # *rect* = [left, bottom, width, height]
        cax = plt.axes([0.85, 0.125, 0.015, 0.75])
        plt.colorbar(cax=cax)

        # show figure
        # plt.show()
        plt.draw()
        plt.pause(0.025)
        # plt.pause(15)

    def save(self, image_file):
        # save image
        # pp = PdfPages(image_file)
        # plt.savefig(pp, format='pdf')
        # pp.close()
        self.fig.savefig(image_file, dpi=75)

    def close(self):
        # close plot GUI
        plt.close()


class PlotDynamicalMatrix4NGram:
    def __init__(self, matrix_input, matrix_output, matrix_predict):
        """
        Initialize the value of matrix.
        :param matrix_list: a goup of matrix.
        :return: non.
        """
        # set figure size
        self.fig = plt.figure(figsize=(20.5, 1.5))
        # self.fig = plt.figure()

        plt.ion()
        self.update(matrix_input, matrix_output, matrix_predict)

    def update(self, matrix_input, matrix_output, matrix_predict):
        # print(matrix_input[0])
        # print(matrix_output[0])
        # print(matrix_predict[0])
        # matrix = np.zeros((3, len(matrix_input[0])), dtype=np.uint8)
        # matrix[0] = matrix_input[0]
        # matrix[1] = matrix_output[0]
        # matrix[2] = matrix_predict[0]
        matrix = np.zeros((6, len(matrix_input[0])), dtype=np.uint8)
        matrix[0] = matrix_input[0]
        matrix[1] = matrix_input[1]
        matrix[2] = matrix_output[0]
        matrix[3] = matrix_output[1]
        matrix[4] = matrix_predict[0]
        matrix[5] = matrix_predict[1]

        # print(matrix)

        axes_w = plt.gca()
        plt.imshow(matrix, interpolation='none')
        plt.xlabel("$Time \longrightarrow$")
        # plt.ylabel("$w_{2}$")
        # axes_w.set_xticks([])
        axes_w.set_yticks([])
        # plt.title("N Gram")

        # show figure
        # plt.show()
        plt.draw()
        plt.pause(0.025)
        # plt.pause(15)

    def save(self, image_file):
        # save image
        # pp = PdfPages(image_file)
        # plt.savefig(pp, format='pdf')
        # pp.close()
        self.fig.savefig(image_file, dpi=75)

    def close(self):
        # close plot GUI
        plt.close()



class PlotDynamicalMatrix4PrioritySort:
    def __init__(self, matrix_input, matrix_output, matrix_predict):
        """
        Initialize the value of matrix.
        :param matrix_list: a goup of matrix.
        :return: non.
        """
        # set figure size
        self.fig = plt.figure(figsize=(6, 5))
        # self.fig = plt.figure()

        plt.ion()
        self.update(matrix_input, matrix_output, matrix_predict)

    def update(self, matrix_input, matrix_output, matrix_predict):

        # draw first line
        axes_input = plt.subplot2grid((3, 1), (0, 0), colspan=1)
        axes_input.set_aspect('equal')
        plt.imshow(matrix_input, interpolation='none')
        axes_input.set_xticks([])
        axes_input.set_yticks([])

        # draw second line
        axes_output = plt.subplot2grid((3, 1), (1, 0), colspan=1)
        plt.imshow(matrix_output, interpolation='none')
        axes_output.set_xticks([])
        axes_output.set_yticks([])

        # draw third line
        axes_predict = plt.subplot2grid((3, 1), (2, 0), colspan=1)
        plt.imshow(matrix_predict, interpolation='none')
        axes_predict.set_xticks([])
        axes_predict.set_yticks([])
        # for 8bits 20length
        # # add text
        # plt.text(-2, -22, name_list[0], ha='right')
        # plt.text(-2, -9, name_list[1], ha='right')
        # plt.text(-2, 4.5, name_list[2], ha='right')
        # plt.text(12, 12, 'Time $\longrightarrow$', ha='right')
        #
        # title = "Repeat Times = %d"%repeat_times
        # plt.text(60, -30, title, ha='center')
        # # plt.title(title)
        #

        # # add text
        # plt.text(-2, -11.3, "Input", ha='right')
        # plt.text(-2, -4.8, "Output", ha='right')
        # plt.text(-2, 2, "Predict", ha='right')
        # plt.text(5.5, 6, 'Time $\longrightarrow$', ha='right')
        # # plt.title(title)

        # set tick labels invisible
        make_tick_labels_invisible(plt.gcf())
        # adjust spaces
        plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.1, right=0.8, top=0.9)
        # add color bars
        # *rect* = [left, bottom, width, height]
        cax = plt.axes([0.85, 0.125, 0.015, 0.75])
        plt.colorbar(cax=cax)

        # show figure
        # plt.show()
        plt.draw()
        plt.pause(0.025)
        # plt.pause(1)


    def save(self, image_file):
        # save image
        # pp = PdfPages(image_file)
        # plt.savefig(pp, format='pdf')
        # pp.close()
        self.fig.savefig(image_file, dpi=75)

    def close(self):
        # close plot GUI
        plt.close()
