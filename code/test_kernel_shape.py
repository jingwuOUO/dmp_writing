import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc

def kernel_shape():
    x = np.arange(-0.5, 1.5, 1e-4)
    y1 = gaussian(x, 0.03, 0.2)
    y2 = gaussian(x, 0.03, 0.5)
    y3 = gaussian(x, 0.03, 0.8)
    plt.figure(1)
    plt.subplot(131)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.xticks([])
    plt.ylabel(r"$\phi (x) $")

    plt.subplot(132)
    yy1 = tr_gaussian(x, 0.03, 0.2)
    yy2 = tr_gaussian(x, 0.03, 0.5)
    yy3 = tr_gaussian(x, 0.03, 0.8)
    plt.plot(x, yy1)
    plt.plot(x, yy2)
    plt.plot(x, yy3)
    plt.xticks([])
    plt.ylabel(r"$\phi (x) $")
    
    plt.subplot(133)
    yy1 = tr_gaussian1(x, 0.03, 0.2)
    yy2 = tr_gaussian1(x, 0.03, 0.5)
    yy3 = tr_gaussian1(x, 0.03, 0.8)
    plt.plot(x, yy1)
    plt.plot(x, yy2)
    plt.plot(x, yy3)
    plt.xticks([])
    plt.ylabel(r"$\phi (x) $")

    plt.show()

def gaussian(x, sigma, mu):
    # a = 1/(math.sqrt(2 * math.pi * sigma))
    a = 1.0
    b = -1/(2 * sigma)
    x = b * (x - mu)**2
    x = np.exp(x)
    y = a * x
    return y

def tr_gaussian(x, sigma, mu):
    a = 1.0
    b = -1/(2 * sigma)
    xx = b * (x - mu)**2
    xx = np.exp(xx)
    y = a * xx
    yy = np.where(abs(x-mu)<0.15, y, 0)
    return yy

def tr_gaussian1(x, sigma, mu):
    a = 1.0
    b = -1/(2 * sigma)
    xx = b * (x - mu)**2
    xx = np.exp(xx)
    y = a * xx
    yy = np.where(abs(x-mu)<0.10, y, 0)
    return yy

if __name__ == "__main__":
    kernel_shape()