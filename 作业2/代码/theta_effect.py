import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *
import matplotlib.animation as animation

"""
本程序主要处理发散角的不同带给波形的影响
本实验选取的发散角为0.1pi~0.4pi
本实验扫描长度固定为2cm，波长选用为632.8nm
"""

fig = plt.figure()
ax = fig.add_subplot()

num = 400

# 设置扫描长度(m)
L = 0.02

# 固定发散角
sita = np.linspace(0.1*np.pi, 0.4*np.pi, num)

# 设置波长
lam = 632.8*10**(-9)
sigma0 = 1/lam


sigma = np.arange(sigma0 - 10**5, sigma0 + 10**5, 10)

sigma_jiehe = np.arange(-10**4, 10**4, 1)
print("Length of sigma_jiehe = %d" %len(sigma_jiehe))

# 扫描长度的影响 
Y01 = 2*L*np.sinc(2*np.pi*sigma_jiehe*L)

def conv(i,j):
    # 设置入射发散角的立体角
    W =2*np.pi*(1-np.cos(j))
    sigma1_1 = sigma0 - sigma0*W/(50*np.pi)

    # 入射角的影响
    B1 = np.pi/(sigma0*W)*((sigma>=sigma1_1)&(sigma<=sigma0))
    I1 = np.convolve(B1,Y01,'same') 
    return I1

# 单独考虑发散角的影响
for i in sita:
    ax.plot(sigma_jiehe, conv(L, i),'r')
    plt.xlim(-5000,1000)
    # plt.ylim(-0.5*10**(-6),3*10**(-6))
    print(i)
    title= ax.text(0.5,1.05,"theta = {:.4f} rad".format(i), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
    plt.pause(1e-5)
    ax.cla()

plt.show()


