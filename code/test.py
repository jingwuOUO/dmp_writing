import numpy as np
import matplotlib.pyplot as plt
import math


# x = [20,35,50,75,100,125]
# y = [9.614967500270199, 3.8460926486571543, 2.098464656310872, 1.1034455414081976, 0.8379834116240682, 0.7455066674459885]
# y = [round(i, 3) for i in y]
# plt.plot(x, y, marker="+", color="b")
# for i_x, i_y in zip(x, y):
#     plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
# plt.xlabel('Kernel number')
# plt.ylabel('Sum of euclidean distance/m')
# plt.title('Euclidean distance of original trajectory and trajectory computed by DMP')
# plt.legend(loc='upper right', frameon=False)
# plt.show()

x = np.arange(0,1,0.000001).tolist()
y = [[] for i in range(5)]
c = 0.05

for index in range(5):
    for i in x:
        # if abs(i-c) <= 0.15:
        #     y[index].append(math.exp(-math.pow(i-c, 2)))
        # else:
        #     y[index].append(0)
        y[index].append(math.exp(-math.pow(i-c, 2)/(2*0.2**2)))
    c += 0.2
    plt.plot(x, y[index], color="b")

plt.xlabel('X')
plt.ylabel('Phi(X)')
plt.title('Kernel Shape of Gaussian')
plt.legend(loc='upper right', frameon=False)
plt.show()




