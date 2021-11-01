import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *

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
L = 532*10**(-9)

sigma0 = 1/L
# sigma1_1 = sigma0 - sigma0*W1/(50*np.pi)
# sigma1_2 = sigma0 - sigma0*W2/(50*np.pi)
# sigma1_3 = sigma0 - sigma0*W3/(50*np.pi)
sigma1_1 = sigma0 - sigma0*W1/(100*np.pi)
sigma1_2 = sigma0 - sigma0*W2/(100*np.pi)
sigma1_3 = sigma0 - sigma0*W3/(100*np.pi)
sigma2 = sigma0

# 设置波束范围
sigma = np.arange(sigma0 - 10**5, sigma2 + 10**5)
# sigma_jiehe = np.arange(-10**4, 10**4, 0.1)
sigma_jiehe = np.arange(sigma0 - 10**5, sigma2 + 10**5)

# 入射角的影响
B1 = np.pi/(sigma0*W1)*((sigma>=sigma1_1)&(sigma<=sigma2))
B2 = np.pi/(sigma0*W2)*((sigma>=sigma1_2)&(sigma<=sigma2))
B3 = np.pi/(sigma0*W3)*((sigma>=sigma1_3)&(sigma<=sigma2))

# 扫描长度的影响 
Y01 = 2*L1*np.sinc(2*np.pi*(sigma0-sigma)*L1) 
Y02 = 2*L2*np.sinc(2*np.pi*(sigma0-sigma)*L2) 
Y03 = 2*L3*np.sinc(2*np.pi*(sigma0-sigma)*L3)

# 卷积(长度和角度的影响)
I1=np.convolve(B1,Y01,'same')
I2=np.convolve(B1,Y02,'same')
I3=np.convolve(B1,Y03,'same')
I4=np.convolve(B2,Y01,'same')
I5=np.convolve(B2,Y02,'same') 
I6=np.convolve(B2,Y03,'same') 
I7=np.convolve(B3,Y01,'same') 
I8=np.convolve(B3,Y02,'same') 
I9=np.convolve(B3,Y03,'same')

plt.subplot(3,3,1)
plt.plot(sigma_jiehe,I1)
# plt.xlim(-10000,10000)
#plt.ylim(-0.5*10**(-5),3*10**(-5))
plt.title("532nm-0.01cm-0.1pi", fontsize = 10)

plt.subplot(3,3,2)
plt.plot(sigma_jiehe,I2)
# plt.xlim(-1000,1000)
#plt.ylim(-0.5*10**(-5),3*10**(-5))
plt.title("532nm-0.1cm-0.1pi", fontsize = 10)

plt.subplot(3,3,3)
plt.plot(sigma_jiehe,I3)
plt.xlim(-2000,2000)
#plt.ylim(-0.5*10**(-5),3*10**(-5))
plt.title("532nm-2cm-0.1pi", fontsize = 10)

plt.subplot(3,3,4)
plt.plot(sigma_jiehe,I4)
plt.xlim(-10000,10000)
plt.ylim(-0.5*10**(-5),1*10**(-5))
plt.title("532nm-0.01cm-0.2pi", fontsize = 10)

plt.subplot(3,3,5)
plt.plot(sigma_jiehe,I5)
plt.xlim(-3000,3000)
plt.ylim(-0.5*10**(-5),1*10**(-5))
plt.title("532nm-0.1cm-0.2pi", fontsize = 10)

plt.subplot(3,3,6)
plt.plot(sigma_jiehe,I6)
plt.xlim(-3000,3000)
plt.ylim(-0.5*10**(-5),1*10**(-5))
plt.title("532nm-2cm-0.2pi", fontsize = 10)

plt.subplot(3,3,7)
plt.plot(sigma_jiehe,I7)
plt.xlim(-10000,10000)
plt.ylim(-0.1*10**(-5),0.2*10**(-5))
plt.title("532nm-0.01cm-0.4pi", fontsize = 10)

plt.subplot(3,3,8)
plt.plot(sigma_jiehe,I8)
plt.xlim(-10000,5000)
plt.ylim(-0.1*10**(-5),0.2*10**(-5))
plt.title("532nm-0.1cm-0.4pi", fontsize = 10)

plt.subplot(3,3,9)
plt.plot(sigma_jiehe,I9)
plt.xlim(-10000,5000)
plt.ylim(-0.1*10**(-5),0.2*10**(-5))
plt.title("532nm-2cm-0.4pi", fontsize = 10)

plt.suptitle("532nm - Spectral measurement curve under the influence of two factors", fontsize = 20) 

plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.2, 
                            hspace=0.35)
plt.show()
