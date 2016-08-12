"""
contourf.
"""
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


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


def show_multi_matirix(w_z, w_z_title, u_z, u_z_title, w_r, w_r_title, u_r, u_r_title,
         w_h, w_h_title, u_h, u_h_title):
    """
    Show the weight matrices of GRU.
    :param w_z: the weight matrix W of update gate.
    :param w_z_title: the title of the weight matrix W of update gate.
    :param u_z: the weight matrix U of update gate.
    :param u_z_title: the title of the weight matrix U of update gate.
    :param w_r: the weight matrix W of reset gate.
    :param w_r_title: the title of the weight matrix W of reset gate.
    :param u_r: the weight matrix U of reset gate.
    :param u_r_title: the title of the weight matrix U of reset gate.
    :param w_h: the weight matrix W of hidden cell.
    :param w_h_title: the title of the weight matrix W of hidden.
    :param u_h: the weight matrix U of hidden cell.
    :param u_h_title: the title of the weight matrix U of hidden.
    :return: None.
    """
    # show w_z matrix of update gate.
    axes_w_z = plt.subplot(2, 3, 1)
    plt.imshow(w_z)
    plt.colorbar()
    plt.xlabel("$x_{t}$")
    plt.ylabel("$z_{t}$")
    axes_w_z.set_xticks([])
    axes_w_z.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_z[0]), len(w_z))
    w_z_title += matrix_size
    plt.title(w_z_title)

    # show w_r matrix of reset gate.
    axes_w_r = plt.subplot(2, 3, 2)
    plt.imshow(w_r)
    plt.colorbar()
    plt.xlabel("$x_{t}$")
    plt.ylabel("$r_{t}$")
    axes_w_r.set_xticks([])
    axes_w_r.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_r[0]), len(w_r))
    w_r_title += matrix_size
    plt.title(w_r_title)

    # show w_h matrix of hidden cell.
    axes_w_h = plt.subplot(2, 3, 3)
    plt.imshow(w_h)
    plt.colorbar()
    plt.xlabel("$x_{t}$")
    plt.ylabel("$h_{t}$")
    axes_w_h.set_xticks([])
    axes_w_h.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_h[0]), len(w_h))
    w_h_title += matrix_size
    plt.title(w_h_title)

    # show u_z matrix of update gate.
    axes_u_z = plt.subplot(2, 3, 4)
    plt.imshow(u_z)
    plt.colorbar()
    plt.xlabel("$h_{t}$")
    plt.ylabel("$z_{t}$")
    axes_u_z.set_xticks([])
    axes_u_z.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_z[0]), len(u_z))
    u_z_title += matrix_size
    plt.title(u_z_title)

    # show u_r matrix of reset gate.
    axes_u_r = plt.subplot(2, 3, 5)
    plt.imshow(u_r)
    plt.colorbar()
    plt.xlabel("$h_{t}$")
    plt.ylabel("$r_{t}$")
    axes_u_r.set_xticks([])
    axes_u_r.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_r[0]), len(u_r))
    u_r_title += matrix_size
    plt.title(u_r_title)

    # show u_h matrix of hidden cell.
    axes_u_h = plt.subplot(2, 3, 6)
    plt.imshow(u_h)
    plt.colorbar()
    plt.xlabel("$h_{t}$")
    plt.ylabel("$h_{t}$")
    axes_u_h.set_xticks([])
    axes_u_h.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_h[0]), len(u_h))
    u_h_title += matrix_size
    plt.title(u_h_title)

    # show the six matrices.
    plt.show()


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
    # axes_target_10.set_aspect('equal')
    plt.imshow(target_sequence_10)
    # plt.grid()
    axes_target_20 = plt.subplot2grid((4, 11), (0, 1), colspan=2)
    # axes_target_20.set_aspect('equal')
    plt.imshow(target_sequence_20)
    axes_target_30 = plt.subplot2grid((4, 11), (0, 3), colspan=3)
    # axes_target_30.set_aspect('equal')
    plt.imshow(target_sequence_30)
    axes_target_50 = plt.subplot2grid((4, 11), (0, 6), colspan=5)
    plt.imshow(target_sequence_50)
    # axes_target_50.set_aspect('equal')

    # draw second line
    axes_output_10 = plt.subplot2grid((4, 11), (1, 0), colspan=1)
    plt.imshow(output_sequence_10)
    axes_output_20 = plt.subplot2grid((4, 11), (1, 1), colspan=2)
    plt.imshow(output_sequence_20)
    axes_output_30 = plt.subplot2grid((4, 11), (1, 3), colspan=3)
    plt.imshow(output_sequence_30)
    axes_output_50 = plt.subplot2grid((4, 11), (1, 6), colspan=5)
    plt.imshow(output_sequence_50)

    # draw last two lines
    axes_target_120 = plt.subplot2grid((4, 11), (2, 0), colspan=11)
    plt.imshow(target_sequence_120)
    axes_output_120 = plt.subplot2grid((4, 11), (3, 0), colspan=11)
    plt.imshow(output_sequence_120)

    # add text
    plt.text(-2, 5, 'Outputs', ha='right')
    plt.text(-2, -7.5, 'Targets', ha='right')
    plt.text(-2, -20, 'Outputs', ha='right')
    plt.text(-2, -32.5, 'Targets', ha='right')
    plt.text(10, 12, 'Time $\longrightarrow$', ha='right')

    # set tick labels invisible
    make_ticklabels_invisible(plt.gcf())
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
