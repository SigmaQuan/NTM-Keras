import struct


id_i = 0
value_i = 0.0
id_i_1 = 0
value_i_1 = 1.0
number = 0
with open("sort.dat", "r") as f:
    for line in f:
        line = line.replace('(', '')
        line = line.replace(')', '')
        line = line.replace(' ', '')
        line = line.replace('\n', '')
        # print line
        [id, value] = line.split(',')
        id = int(id)
        id = id*128/1024
        value = float(value)
        # print(id)
        # print(value)

        if (id_i == id):
            value_i = value_i + value
            number = number + 1
        else:
            if (id_i > id_i_1 + 300 or value_i < value_i_1 - 0.09):
                print '({0}, {1})'.format(id, value_i/number)
            id_i_1 = id_i
            value_i_1 = value_i/number
            id_i = id
            value_i = value
            number = 1

    print '({0}, {1})'.format(id_i, value_i/number)



# for line in open("sort.dat").readlines():
#     print line