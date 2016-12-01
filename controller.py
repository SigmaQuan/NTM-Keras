from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class Controller(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Controller, self).__init__(**kwargs)

    def build(self, input_shape, mem_shape, n_heads, hidden_dim):
        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, self.output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
        self.mem_shape = mem_shape
        self.Memory = np.zeros((mem_shape[0], mem_shape[1]))
