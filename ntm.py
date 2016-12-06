from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec

from keras.layers import Recurrent
from keras.layers import time_distributed_dense


class NTM(Recurrent):
    def __init__(self, output_dim, memory_dim=128, memory_size=20,
                 controller_output_dim=100, location_shift_range=1,
                 num_read_head=1, num_write_head=1,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None,
                 b_regularizer=None, W_y_regularizer=None,
                 W_xi_regularizer=None, W_r_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        """
        Neural Turing Machines - Alex Graves et. al, 2014.
            For a step-by-step description of the algorithm, see
            [this paper](https://arxiv.org/pdf/1410.5401.pdf).
        # Arguments
        :param output_dim: dimension of the internal projections and the
        final output.
        :param memory_dim: the dimension of one item in the external memory.
        :param memory_size: the size of total item in the external memory.
        :param controller_dim: the dimension of controller networks.
        :param location_shift_range: the location shift range.
        :param num_read_heads: the number of read heads.
        :param num_write_heads: the number of write heads.
        :param init: weight initialization function.
            Can be the name of an existing function (str), or a Theano
            function (see: [initializations](../initializations.md)).
        :param inner_init: initialization function of the inner cells.
        :param forget_bias_init: initialization function for the bias of
        the forget gate.
            [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        :param activation: activation function.
            Can be the name of an existing function (str),or a Theano
            function (see: [activations](../activations.md)).
        :param inner_activation: activation function for the inner cells.
        :param W_regularizer: instance of [WeightRegularizer].
            (eg. L1 or L2 regularization), applied to the input weights
            matrices.
        :param U_regularizer: instance of [WeightRegularizer].
            (eg. L1 or L2 regularization), applied to the recurrent
            weights matrices.
        :param b_regularizer: instance of [WeightRegularizer] applied to
        the bias.
        :param W_y_regularizer: instance of [WeightRegularizer].
            (eg. L1 or L2 regularization), applied to the output
            weights matrices.
        :param W_xi_regularizer: instance of [WeightRegularizer].
            (eg. L1 or L2 regularization), applied to the interface
            parameters weights matrices.
        :param W_r_regularizer: instance of [WeightRegularizer].
            (eg. L1 or L2 regularization), applied to the read content
            weights matrices.
        :param dropout_W: float between 0 and 1. Fraction of the input
        units to drop for input gates.
        :param dropout_U: float between 0 and 1. Fraction of the input
        units to drop for recurrent connections.
        :param kwargs: non.

        # References
            - [Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf)
            - [Hybrid computing using a neural network with dynamic
              external memory]
            (http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html)
            - [One-shot learning with memory-augmented neural networks]
              (https://arxiv.org/pdf/1605.06065.pdf)
            - [Scaling memory-augmented neural networks with sparse reads
              and writes]
              (https://arxiv.org/pdf/1610.09027.pdf)
        """
        self.output_dim = output_dim
        # add by Robot Steven ********************************************#
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.controller_output_dim = controller_output_dim
        self.location_shift_range = location_shift_range
        self.num_read_head = num_read_head
        self.num_write_head = num_write_head
        # add by Robot Steven ********************************************#
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        # add by Robot Steven ********************************************#
        self.W_y_regularizer = regularizers.get(W_y_regularizer)
        self.W_xi_regularizer = regularizers.get(W_xi_regularizer)
        self.W_r_regularizer = regularizers.get(W_r_regularizer)
        # add by Robot Steven ********************************************#
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(NTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            # self.states = [None, None]  ## commented by Robot Steven
            # add by Robot Steven ****************************************#
            # Comparing with the old self.states, there are four
            # additional item which represents external memory, writing
            # addressing, reading addressing and read_content correspondingly.
            # h_tm1, c_tm1, B_U, B_W, M_tm1, w_w_tm1, w_r_tm1, r_tm1_list
            self.states = [None, None, None, None, None, None, None, None]
            # add by Robot Steven ****************************************#

        if self.consume_less == 'gpu':
            # self.W = self.init((self.input_dim, 4 * self.output_dim),
            #                    name='{}_W'.format(self.name))
            # self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
            #                          name='{}_U'.format(self.name))
            #
            # self.b = K.variable(np.hstack((np.zeros(self.output_dim),
            #                                K.get_value(self.forget_bias_init(
            #                                    (self.output_dim,))),
            #                                np.zeros(self.output_dim),
            #                                np.zeros(self.output_dim))),
            #                     name='{}_b'.format(self.name))
            # self.trainable_weights = [self.W, self.U, self.b]

            # add by Robot Steven ****************************************#
            self.W = self.init((self.input_dim,
                                4 * self.controller_output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.controller_output_dim,
                                      4 * self.controller_output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(
                np.hstack((np.zeros(self.controller_output_dim),
                           K.get_value(self.forget_bias_init(
                               (self.controller_output_dim,))),
                           np.zeros(self.controller_output_dim),
                           np.zeros(self.controller_output_dim))),
                name='{}_b'.format(self.name))
            self.W_y = self.init((self.controller_output_dim,
                                  self.output_dim),
                                 name='{}_W_y'.format(self.name))
            self.W_xi = self.init(
                (self.controller_output_dim,
                 self.num_read_head *
                    # k_{r}, \beta_{r}, g_{r}, s_{r}, \gama_{r}
                    (self.memory_dim + 1 + 1 +
                     (self.location_shift_range*2+1) + 1)
                 +
                 self.num_write_head *
                    # k_{w}, \beta_{w}, g_{w}, s_{w}, \gama_{w}, e_{w}, a_{w}
                    (self.memory_dim + 1 + 1 +
                     (self.location_shift_range*2+1) + 1 +
                     self.memory_dim + self.memory_dim)
                 ),
                name='{}_W_xi'.format(self.name))
            self.W_r = self.init((self.num_read_head * self.memory_dim,
                                  self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.trainable_weights = [self.W, self.U, self.b, self.W_y,
                                      self.W_xi, self.W_r]
            # add by Robot Steven ****************************************#
        else:
            # self.W_i = self.init((self.input_dim, self.output_dim),
            #                      name='{}_W_i'.format(self.name))
            # self.U_i = self.inner_init((self.output_dim, self.output_dim),
            #                            name='{}_U_i'.format(self.name))
            # self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))
            #
            # self.W_f = self.init((self.input_dim, self.output_dim),
            #                      name='{}_W_f'.format(self.name))
            # self.U_f = self.inner_init((self.output_dim, self.output_dim),
            #                            name='{}_U_f'.format(self.name))
            # self.b_f = self.forget_bias_init((self.output_dim,),
            #                                  name='{}_b_f'.format(self.name))
            #
            # self.W_c = self.init((self.input_dim, self.output_dim),
            #                      name='{}_W_c'.format(self.name))
            # self.U_c = self.inner_init((self.output_dim, self.output_dim),
            #                            name='{}_U_c'.format(self.name))
            # self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))
            #
            # self.W_o = self.init((self.input_dim, self.output_dim),
            #                      name='{}_W_o'.format(self.name))
            # self.U_o = self.inner_init((self.output_dim, self.output_dim),
            #                            name='{}_U_o'.format(self.name))
            # self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))
            #
            # self.trainable_weights = [self.W_i, self.U_i, self.b_i,
            #                           self.W_c, self.U_c, self.b_c,
            #                           self.W_f, self.U_f, self.b_f,
            #                           self.W_o, self.U_o, self.b_o]
            #
            # self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            # self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            # self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

            # add by Robot Steven ****************************************#
            self.W_i = self.init(
                (self.input_dim, self.controller_output_dim),
                name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init(
                (self.controller_output_dim, self.controller_output_dim),
                name='{}_U_i'.format(self.name))
            self.b_i = K.zeros(
                (self.controller_output_dim,),
                name='{}_b_i'.format(self.name))

            self.W_f = self.init(
                (self.input_dim, self.controller_output_dim),
                name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init(
                (self.controller_output_dim, self.controller_output_dim),
                name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init(
                (self.controller_output_dim,),
                name='{}_b_f'.format(self.name))

            self.W_c = self.init(
                (self.input_dim, self.controller_output_dim),
                name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init(
                (self.controller_output_dim, self.controller_output_dim),
                name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.controller_output_dim,),
                               name='{}_b_c'.format(self.name))

            self.W_o = self.init(
                (self.input_dim, self.controller_output_dim),
                name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init(
                (self.controller_output_dim, self.controller_output_dim),
                name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.controller_output_dim,),
                               name='{}_b_o'.format(self.name))

            self.W_y = self.init((self.controller_output_dim,
                                  self.output_dim),
                                 name='{}_W_y'.format(self.name))

            self.W_xi_k_r = self.init(
                (self.controller_output_dim,
                 self.num_read_head * self.memory_dim),
                name='{}_W_xi_k_r'.format(self.name))
            self.W_xi_beta_r = self.init(
                (self.controller_output_dim,
                 self.num_read_head * 1),
                name='{}_W_xi_beta__r'.format(self.name))
            self.W_xi_g_r = self.init(
                (self.controller_output_dim,
                 self.num_read_head * 1),
                name='{}_W_xi_g_r'.format(self.name))
            self.W_xi_s_r = self.init(
                (self.controller_output_dim,
                 self.num_read_head * (self.location_shift_range * 2 + 1)),
                name='{}_W_xi_s_r'.format(self.name))
            self.W_xi_gama_r = self.init(
                (self.controller_output_dim,
                 self.num_read_head * 1),
                name='{}_W_xi_gama_r'.format(self.name))

            self.W_xi_k_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * self.memory_dim),
                name='{}_W_xi_k_w'.format(self.name))
            self.W_xi_beta_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * 1),
                name='{}_W_xi_beta__w'.format(self.name))
            self.W_xi_g_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * 1),
                name='{}_W_xi_g_w'.format(self.name))
            self.W_xi_s_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * (self.location_shift_range * 2 + 1)),
                name='{}_W_xi_s_w'.format(self.name))
            self.W_xi_gama_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * 1),
                name='{}_W_xi_gama_w'.format(self.name))
            self.W_xi_e_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * self.memory_dim),
                name='{}_W_xi_e_w'.format(self.name))
            self.W_xi_a_w = self.init(
                (self.controller_output_dim,
                 self.num_write_head * self.memory_dim),
                name='{}_W_xi_a_w'.format(self.name))

            self.W_r = self.init((self.num_read_head * self.memory_dim,
                                  self.output_dim),
                                 name='{}_W_r'.format(self.name))

            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o,
                                      self.W_y,
                                      self.W_xi_k_r, self.W_xi_beta_r,
                                      self.W_xi_g_r, self.W_xi_s_r,
                                      self.W_xi_gama_r,
                                      self.W_xi_k_w, self.W_xi_beta_w,
                                      self.W_xi_g_w, self.W_xi_s_w,
                                      self.W_xi_gama_w,
                                      self.W_xi_e_w, self.W_xi_a_w,
                                      self.W_r]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])
            self.W_xi = K.concatenate(
                [self.W_xi_k_r, self.W_xi_beta_r, self.W_xi_g_r,
                 self.W_xi_s_r, self.W_xi_gama_r,
                 self.W_xi_k_w, self.W_xi_beta_w, self.W_xi_g_w,
                 self.W_xi_s_w, self.W_xi_gama_w,
                 self.W_xi_e_w, self.W_xi_a_w])
            # add by Robot Steven ****************************************#

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        # add by Robot Steven ********************************************#
        if self.W_y_regularizer:
            self.b_regularizer.set_param(self.W_y)
        if self.W_xi_regularizer:
            self.W_xi_regularizer.set_param(self.W_xi)
        if self.W_r_regularizer:
            self.W_r_regularizer.set_param(self.W_r_regularizer)
        # add by Robot Steven ********************************************#

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            # K.set_value(self.states[0],
            #             np.zeros((input_shape[0], self.output_dim)))
            # K.set_value(self.states[1],
            #             np.zeros((input_shape[0], self.output_dim)))
            # add by Robot Steven ****************************************#
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.controller_output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.controller_output_dim)))
            # add by Robot Steven ****************************************#
        else:
            # self.states = [K.zeros((input_shape[0], self.output_dim)),
            #                K.zeros((input_shape[0], self.output_dim))]
            # add by Robot Steven ****************************************#
            self.states = [K.zeros((input_shape[0], self.output_dim)),  # h_tm1
                           K.zeros((input_shape[0], self.output_dim))]  # c_tm1
                           # K.zeros((input_shape[0], self.output_dim)),
                           # K.zeros((input_shape[0], self.output_dim)),
                           # K.zeros((input_shape[0], self.output_dim)),
                           # K.zeros((input_shape[0], self.output_dim))]
            # add by Robot Steven ****************************************#

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            # x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
            #                              input_dim, self.output_dim, timesteps)
            # x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
            #                              input_dim, self.output_dim, timesteps)
            # x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
            #                              input_dim, self.output_dim, timesteps)
            # x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
            #                              input_dim, self.output_dim, timesteps)
            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.controller_output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.controller_output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.controller_output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.controller_output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        M_tm1 = states[4]
        w_w_tm1 = states[5]
        w_r_tm1 = states[6]
        r_tm1_list = states[7]

        if self.consume_less == 'gpu':
            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

            # z0 = z[:, :self.output_dim]
            # z1 = z[:, self.output_dim: 2 * self.output_dim]
            # z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            # z3 = z[:, 3 * self.output_dim:]
            z0 = z[:, :self.controller_output_dim]
            z1 = z[:, self.controller_output_dim: 2 * self.controller_output_dim]
            z2 = z[:, 2 * self.controller_output_dim: 3 * self.controller_output_dim]
            z3 = z[:, 3 * self.controller_output_dim:]

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'cpu':
                # x_i = x[:, :self.output_dim]
                # x_f = x[:, self.output_dim: 2 * self.output_dim]
                # x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                # x_o = x[:, 3 * self.output_dim:]
                x_i = x[:, :self.controller_output_dim]
                x_f = x[:, self.controller_output_dim: 2 * self.controller_output_dim]
                x_c = x[:, 2 * self.controller_output_dim: 3 * self.controller_output_dim]
                x_o = x[:, 3 * self.controller_output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'memory_dim': self.memory_dim,
                  'memory_size': self.memory_size,
                  'controller_output_dim': self.controller_output_dim,
                  'location_shift_range': self.location_shift_range,
                  'num_read_head': self.num_read_head,
                  'num_write_head': self.num_write_head,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'W_y_regularizer': self.W_y_regularizer.get_config() if self.W_y_regularizer else None,
                  'W_xi_regularizer': self.W_xi_regularizer.get_config() if self.W_xi_regularizer else None,
                  'W_r_regularizer': self.W_r_regularizer.get_config() if self.W_r_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(NTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
