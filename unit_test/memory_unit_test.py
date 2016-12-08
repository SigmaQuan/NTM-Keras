import memory
import head
# import write_heads

from keras import backend as K

number_of_memory_locations = 6
memory_vector_size = 3

memory_t = memory.initial(number_of_memory_locations, memory_vector_size)

weight_t = K.random_binomial((number_of_memory_locations, 1), 0.2)

read_vector = head.reading(memory_t, weight_t)

print memory_t.shape
print weight_t.shape
print read_vector

