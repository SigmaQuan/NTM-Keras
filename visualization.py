"""
Visualization of Neural Turing Machines.
"""
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


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


def show_copy_data(target_sequence_10, output_sequence_10,
                   target_sequence_20, output_sequence_20,
                   target_sequence_30, output_sequence_30,
                   target_sequence_50, output_sequence_50,
                   target_sequence_120, output_sequence_120,
                   image_file):
    # set figure size
    plt.figure(figsize=(12, 4))

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
    pp = PdfPages(image_file)
    plt.savefig(pp, format='pdf')
    pp.close()

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

