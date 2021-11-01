"""
本程序主要模拟扫描长度与发散角的不同带给波形的影响
本实验选取的扫描长度为0.01cm，0.5cm与2cm
本实验发散角选取为0.1pi，0.2pi与0.4pi，波长选用为632.8nm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *
import matplotlib.animation as animation


def fft_convolve(a,b):
    n = len(a) + len(b) - 1
    N = 2**(int(np.log2(n))+1)
    A = fft(a, N)
    B = fft(b, N)
    return ifft(A*B)[:n]

N = 200000

# 设置扫描长度(m)
L1 = 0.0001
L2 = 0.001
L3 = 0.02

# 设置三个入射发散角
sita1=0.1*np.pi 
sita2=0.2*np.pi
sita3=0.4*np.pi

# 设置三个入射发散角的立体角
W1=2*np.pi*(1-np.cos(sita1))
W2=2*np.pi*(1-np.cos(sita2))
W3=2*np.pi*(1-np.cos(sita3))

# 设置波长
L = 632.8*10**(-9)

sigma0 = 1/L
sigma1_1 = sigma0 - sigma0*W1/(100*np.pi)
sigma1_2 = sigma0 - sigma0*W2/(100*np.pi)
sigma1_3 = sigma0 - sigma0*W3/(100*np.pi)
sigma2 = sigma0

# 设置波束范围
sigma = np.arange(sigma0 - 10**5, sigma2 + 10**5)
# sigma_jiehe = np.arange(-10**4, 10**4, 0.1)
sigma_jiehe = np.arange(sigma0-10**5, sigma2+10**5)

# 入射角的影响
B1 = np.pi/(sigma0*W1)*((sigma>=sigma1_1)&(sigma<=sigma2))
B2 = np.pi/(sigma0*W2)*((sigma>=sigma1_2)&(sigma<=sigma2))
B3 = np.pi/(sigma0*W3)*((sigma>=sigma1_3)&(sigma<=sigma2))
print(len(B3))

# 扫描长度的影响 
Y01 = 2*L1*np.sinc(2*np.pi*(sigma0-sigma)*L1) 
Y02 = 2*L2*np.sinc(2*np.pi*(sigma0-sigma)*L2) 
Y03 = 2*L3*np.sinc(2*np.pi*(sigma0-sigma)*L3)
print(len(Y03))


sigma_jiehe1 = np.linspace(sigma0-10**5, sigma2+10**5, len(B3)+len(Y03)-1)

subplot(3,3,1)
plt.plot(sigma_jiehe1,fft_convolve(B1, Y01))
plt.xlim(1.57*10**6, 1.59*10**6)
plt.title("632.8nm-0.01cm-0.1pi", fontsize = 10)

subplot(3,3,2)
plt.plot(sigma_jiehe1,fft_convolve(B1, Y02))
plt.xlim(1.57*10**6, 1.59*10**6)
plt.title("632.8nm-0.1cm-0.2pi", fontsize = 10)

subplot(3,3,3)
plt.plot(sigma_jiehe1,fft_convolve(B1, Y03))
plt.xlim(1.57*10**6, 1.59*10**6)
plt.title("632.8nm-2cm-0.4pi", fontsize = 10)

subplot(3,3,4)
plt.plot(sigma_jiehe1,fft_convolve(B2, Y01))
plt.xlim(1.57*10**6, 1.59*10**6)
plt.title("632.8nm-0.01cm-0.1pi", fontsize = 10)

subplot(3,3,5)
plt.plot(sigma_jiehe1,fft_convolve(B2, Y02))
plt.xlim(1.57*10**6, 1.59*10**6)
plt.title("632.8nm-0.1cm-0.2pi", fontsize = 10)

subplot(3,3,6)
plt.plot(sigma_jiehe1,fft_convolve(B2, Y03))
plt.xlim(1.57*10**6, 1.59*10**6)
plt.title("632.8nm-2cm-0.4pi", fontsize = 10)

subplot(3,3,7)
plt.plot(sigma_jiehe1,fft_convolve(B3, Y01))
plt.xlim(1.56*10**6, 1.59*10**6)
plt.title("632.8nm-0.01cm-0.1pi", fontsize = 10)

subplot(3,3,8)
plt.plot(sigma_jiehe1,fft_convolve(B3, Y02))
plt.xlim(1.56*10**6, 1.59*10**6)
plt.title("632.8nm-0.1cm-0.2pi", fontsize = 10)

subplot(3,3,9)
plt.plot(sigma_jiehe1,fft_convolve(B3, Y03))
plt.xlim(1.56*10**6, 1.59*10**6)
plt.title("632.8nm-2cm-0.4pi", fontsize = 10)

plt.suptitle("632.8nm - Spectral measurement curve under the influence of two factors", fontsize = 20) 
plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.2, 
                            hspace=0.35)
# plt.show()

I0 = fft_convolve(B1, Y01)
y=max(I0)-min(I0)
print(max(I0))
print(min(I0))
y0=np.where(I0==abs(y/2))
print(y0)
