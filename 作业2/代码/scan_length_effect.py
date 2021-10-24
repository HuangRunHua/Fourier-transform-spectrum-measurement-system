"""
本程序主要处理扫描长度的不同带给波形的影响
本实验选取的扫描长度为0.01cm~2cm并按0.01cm选取200个点
本实验发散角固定为0.4pi，波长选用为632.8nm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot()

num = 200

# 设置扫描长度(m)
L = np.linspace(0.0001, 0.02, num)

# 固定发散角
sita1=0.4*np.pi

# 设置入射发散角的立体角
W =2*np.pi*(1-np.cos(sita1))

# 设置波长
lam = 632.8*10**(-9)
sigma0 = 1/lam
sigma1_1 = sigma0 - sigma0*W/(50*np.pi)

sigma = np.arange(sigma0 - 10**5, sigma0 + 10**5, 10)

sigma_jiehe = np.arange(-10**4, 10**4, 0.1)
print("Length of sigma_jiehe = %d" %len(sigma_jiehe))


def conv(i,j):
    # 入射角的影响
    B1 = np.pi/(sigma0*W)*((sigma>=sigma1_1)&(sigma<=sigma0))

    # 扫描长度的影响 
    Y01 = 2*i*np.sinc(2*np.pi*sigma_jiehe*i) 
    I1 = np.convolve(B1,Y01,'same') 
    return I1

# 单独考虑扫描长度的影响
for i in L:
    ax.plot(sigma_jiehe, conv(i,sita1),'r')
    plt.xlim(-500,100)
    plt.ylim(-0.5*10**(-6),3*10**(-6))
    print(i)
    title= ax.text(0.5,1.05,"L = {:.4f}m".format(i), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
    plt.pause(1e-5)
    ax.cla()

plt.show()


