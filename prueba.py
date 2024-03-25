import numpy

a = [1,2,3]
b = [4,5,6]
c = [7, 8, 9]

total = []
total.append(a)
total.append(b)
total.append(c)

media = numpy.mean([a,b, c], axis = 0)
desv = numpy.std(total, axis = 0)
print(desv)

result = media + desv
print(result)