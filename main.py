import struct

#
# id_i = 0
# value_i = 0.0
# id_i_1 = 0
# value_i_1 = 1.0
# number = 0
# with open("sort.dat", "r") as f:
#     for line in f:
#         line = line.replace('(', '')
#         line = line.replace(')', '')
#         line = line.replace(' ', '')
#         line = line.replace('\n', '')
#         # print line
#         id_value = line.split(',')
#
#         if len(id_value) < 2:
#             continue
#         # print id_value
#
#         id = int(id_value[0])
#         id = id*128/1024
#         value = float(id_value[1])
#         # print(id)
#         # print(value)
#
#         if (id_i == id):
#             value_i = value_i + value
#             number = number + 1
#         elif (id < 2000):
#
#             if (value_i/number < (value_i_1 - 0.009) and id_i < 100) or ((id_i > (id_i_1 + 20)) and id_i > 100) :
#                 print '({0}, {1})'.format(id_i_1+1, value_i_1)
#                 id_i_1 = id_i
#                 value_i_1 = value_i/number
#
#             id_i = id
#             value_i = value
#             number = 1
#
#
#
# id_i = 0
# value_i = 0.0
# id_i_1 = 0
# value_i_1 = 1.0
# number = 0
# with open("recall.dat", "r") as f:
#     for line in f:
#         line = line.replace('(', '')
#         line = line.replace(')', '')
#         line = line.replace(' ', '')
#         line = line.replace('\n', '')
#         # print line
#         id_value = line.split(',')
#
#         if len(id_value) < 2:
#             continue
#         # print id_value
#
#         id = int(id_value[0])
#         # id = id*128/1024
#         value = float(id_value[1])
#         # print(id)
#         # print(value)
#
#         id_i = id
#         value_i = value
#
#         if (value_i < (value_i_1 - 0.03) and id_i < 100) or ((id_i > (id_i_1 + 40)) and id_i > 100):
#             print '({0}, {1})'.format(id_i_1, value_i_1)
#             id_i_1 = id_i
#             value_i_1 = value_i
#
#

#
# id_i = 0
# value_i = 0.0
# id_i_1 = 0
# value_i_1 = 1.0
# number = 0
# with open("copy.dat", "r") as f:
#     for line in f:
#         line = line.replace('(', '')
#         line = line.replace(')', '')
#         line = line.replace(' ', '')
#         line = line.replace('\n', '')
#         # print line
#         id_value = line.split(',')
#
#         if len(id_value) < 2:
#             continue
#         # print id_value
#
#         id = int(id_value[0])
#         id = id*128/1024
#         value = float(id_value[1])
#         # print(id)
#         # print(value)
#
#         if (id_i == id):
#             value_i = value_i + value
#             number = number + 1
#         elif (id < 2000):
#
#             if (value_i/number < (value_i_1 - 0.009) and id_i < 100) or ((id_i > (id_i_1 + 20)) and id_i > 100) :
#                 print '({0}, {1})'.format(id_i_1+1, value_i_1)
#                 id_i_1 = id_i
#                 value_i_1 = value_i/number
#
#             id_i = id
#             value_i = value
#             number = 1


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
# N = np.arange(100, 4000, 20)
# p = np.arange(0.01, 0.99, 0.02)
# N, p = np.meshgrid(N, p)
# # R = np.sqrt(N ** 2 + p ** 2)
# # D = np.sin(R)
# D = N/p
#
# ax.plot_surface(N, p, D, rstride=1, cstride=1, cmap=plt.cm.hot)
# ax.contourf(N, p, D, zdir='z', offset=-2, cmap=plt.cm.hot)
# # ax.set_zlim(-2, 2)
#
# # savefig('../figures/plot3d_ex.png',dpi=48)
# plt.show()


# set random values of dependency matrix
import numpy as np

tau = 16 + 1

dependency_matrix = np.ones(shape=(tau, tau), dtype=int)
print(dependency_matrix)

for y in range(tau-1, 0, -1):
    for x in range(1, y+1):
        dependency_matrix[x][y] = np.random.randint(0, 2)

print(dependency_matrix)


for x in xrange(tau):
    # print("\% x = %d" % x)
    for y in xrange(tau):
        if x > 0 and y > 0:
            print('(%d, %d) [%d]' % (x, y, dependency_matrix[x][y]))
    print('\n')
