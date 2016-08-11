"""
contourf.
"""
import numpy as np
import visualization

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


if __name__ == "__main__":
    test_show_matrix()
    test_show_multi_matrix()

