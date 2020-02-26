import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc

def kernel_shape():
    import seaborn as sns
    sns.set()
    # sns.axes_style("white")
    # plt.set_figheight(15)
    # plt.set_figwidth(15)

    f = plt.figure(figsize=(10,7))
    plt.ylabel(r"$\phi (x) $")
    plt.xticks([])
    plt.yticks([])
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(223)
    ax4 = f.add_subplot(224)

    x = np.arange(-0.5, 1.5, 1e-4)
    y1 = gaussian(x, 0.03, 0.2)
    y2 = gaussian(x, 0.03, 0.5)
    y3 = gaussian(x, 0.03, 0.8)
    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.plot(x, y3)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xlabel("(a)")
    
    # fig.suptitle('Gaussian Kernel', fontsize=6)

    yy1 = tr_gaussian(x, 0.03, 0.2)
    yy2 = tr_gaussian(x, 0.03, 0.5)
    yy3 = tr_gaussian(x, 0.03, 0.8)
    ax2.plot(x, yy1)
    ax2.plot(x, yy2)
    ax2.plot(x, yy3)
    ax2.set_xticks(np.arange(0, 1.2, 0.2))
    ax2.set_yticks(np.arange(0, 1.2, 0.2))
    ax2.set_xlabel("(b)")
    # ax2.ylabel(r"$\phi (x) $")
    # fig.suptitle('Truncated Gaussian Kernel', fontsize=6)
    
    yy1 = tr_gaussian1(x, 0.03, 0.2)
    yy2 = tr_gaussian1(x, 0.03, 0.5)
    yy3 = tr_gaussian1(x, 0.03, 0.8)
    ax3.plot(x, yy1)
    ax3.plot(x, yy2)
    ax3.plot(x, yy3)
    ax3.set_xticks(np.arange(0, 1.2, 0.2))
    ax3.set_yticks(np.arange(0, 1.2, 0.2))
    ax3.set_xlabel("(c)")
    # ax3.ylabel(r"$\phi (x) $")
    # fig.suptitle('Truncated Gaussian Kernel', fontsize=6)

    yy1 = tr_gaussian1(x, 0.03, 0.4)
    yy2 = tr_gaussian1(x, 0.03, 0.5)
    yy3 = tr_gaussian1(x, 0.03, 0.6)
    ax4.plot(x, yy1)
    ax4.plot(x, yy2)
    ax4.plot(x, yy3)
    ax4.set_xticks(np.arange(0, 1.2, 0.2))
    ax4.set_yticks(np.arange(0, 1.2, 0.2))
    ax4.set_xlabel("(d)")
    # ax4.ylabel(r"$\phi (x) $")
    # fig.suptitle('Truncated Gaussian Kernel', fontsize=6)


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