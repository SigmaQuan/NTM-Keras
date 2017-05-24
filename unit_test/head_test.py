# import keras.backend as K
#
# kvar = K.zeros((3, 54))
# print('dimension: %d' % K.ndim(kvar))
# print('data type: %s' % K.dtype(kvar))
# print('total parameters: %d' % K.count_params(kvar))
#
# x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)  #
# y = K.ones((4, 3, 5))
# dot_x_y = K.dot(x, y)
# print(K.int_shape(x))
# print(K.int_shape(y))
# print(K.int_shape(K.zeros_like(dot_x_y)))
# # print('(2, 3) * (4, 3, 5) -> ', K.int_shape(dot_x_y))

import babel
