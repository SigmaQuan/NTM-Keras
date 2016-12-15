from keras import backend as K
import theano.tensor as T
import theano
import memory
import head
#
# number_of_memory_locations = 6
# memory_vector_size = 3
#
# memory_t = memory.initial(number_of_memory_locations, memory_vector_size)
#
# weight_t = K.random_binomial((number_of_memory_locations, 1), 0.2)
#
# read_vector = head.reading(memory_t, weight_t)
#
# print memory_t.shape
# print weight_t.shape
# print read_vector
#


import numpy as np

u = np.random.random((3))
V = np.random.random((3, 4))
similairty = np.dot(u, V) / (sum(abs(u)) * np.sum(abs(V), axis=0))
print("u")
print(u)
print("V")
print(V)
print("similairty")
print(similairty)
