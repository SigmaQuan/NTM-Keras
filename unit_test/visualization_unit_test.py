"""
contourf.
"""
import numpy as np
import visualization
import dataset


def test_show_matrix():
    w = np.random.random((8, 10))
    title = " "
    visualization.show(w, title)


def test_show_multi_matrix():
    w = np.random.random((200, 300))
    w_z = np.random.random((200, 300))
    u_z = np.random.random((200, 250))
    w_r = w
    u_r = w
    w_h = w
    u_h = w
    w_z_title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$\n $W^{(z)}$"
    u_z_title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$\n $U^{(z)}$"
    w_r_title = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+U^{(r)}h_{t-1}+b^{(r)})$\n $W^{(r)}$"
    u_r_title = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+U^{(r)}h_{t-1}+b^{(r)})$\n $U^{(r)}$"
    w_h_title = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+U(r_{t}\odot h_{t-1})+b^{(h)})$\n $W$"
    u_h_title = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+U(r_{t}\odot h_{t-1})+b^{(h)})$\n $U$"
    visualization.show_multi_matirix(w_z, w_z_title, u_z, u_z_title, w_r, w_r_title, u_r, u_r_title,
         w_h, w_h_title, u_h, u_h_title)


def test_show_copy_data():
    input_sequence_10, output_sequence_10 = dataset.generate_copy_data(8, 10)
    input_sequence_20, output_sequence_20 = dataset.generate_copy_data(8, 20)
    input_sequence_30, output_sequence_30 = dataset.generate_copy_data(8, 30)
    input_sequence_50, output_sequence_50 = dataset.generate_copy_data(8, 50)
    input_sequence_120, output_sequence_120 = dataset.generate_copy_data(8, 120)

    input_sequence_10 = input_sequence_10.transpose()
    output_sequence_10 = output_sequence_10.transpose()

    input_sequence_20 = input_sequence_20.transpose()[:, 0:input_sequence_20.size/2]
    output_sequence_20 = output_sequence_20.transpose()[:, 0:output_sequence_20.size/2]

    input_sequence_30 = input_sequence_30.transpose()[:, 0:input_sequence_30.size/2]
    output_sequence_30 = output_sequence_30.transpose()[:, 0:output_sequence_30.size/2]

    input_sequence_50 = input_sequence_50.transpose()[:, 0:input_sequence_50.size/2]
    output_sequence_50 = output_sequence_50.transpose()[:, 0:output_sequence_50.size/2]

    input_sequence_120 = input_sequence_120.transpose()[:, 0:input_sequence_120.size/2]
    output_sequence_120 = output_sequence_120.transpose()[:, 0:output_sequence_120.size/2]

    print input_sequence_10
    # print output_sequence_10
    print input_sequence_10.size
    print (input_sequence_10.shape[1]-1)/2

    visualization.show_copy_data(
        input_sequence_10[:, 0:(input_sequence_10.shape[1]-1)/2],
        output_sequence_10[:, 0:(output_sequence_10.shape[1]-1)/2],
        input_sequence_20[:, 0:(input_sequence_20.shape[1]-1)/2],
        output_sequence_20[:, 0:(output_sequence_20.shape[1]-1)/2],
        input_sequence_30[:, 0:(input_sequence_30.shape[1]-1)/2],
        output_sequence_30[:, 0:(output_sequence_30.shape[1]-1)/2],
        input_sequence_50[:, 0:(input_sequence_50.shape[1]-1)/2],
        output_sequence_50[:, 0:(output_sequence_50.shape[1]-1)/2],
        input_sequence_120[:, 0:(input_sequence_120.shape[1]-1)/2],
        output_sequence_120[:, 0:(output_sequence_120.shape[1]-1)/2])


if __name__ == "__main__":
    # test_show_matrix()
    # test_show_multi_matrix()
    test_show_copy_data()

    # import matplotlib.pyplot as plt
    # nx = 3
    # ny = 1
    # dxs = 8.0
    # dys = 6.0
    # fig, ax = plt.subplots(ny, nx, sharey=True, figsize=(dxs*nx, dys*ny))
    # for i in range(nx):
    #     ax[i].plot([1, 2], [1, 2])
    # fig.show()

