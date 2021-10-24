import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *

fs=1000
N=300
N1 = 500
t = np.linspace(0,0.3,N)
i=np.cos(2*np.pi*4*t)+np.cos(2*np.pi*16*t)

# # 快速傅立叶变换
y = 2*np.abs(fft(i,500))/N
f = fs*np.arange(N1)/N1

# 快速傅立叶变换
# y = 2*np.abs(fft(i))/N
# f = fs*np.arange(N)/N


plt.subplot(2,1,1)
plt.plot(f, y)
plt.title("Original spectrum")
plt.xlim(0, 20)
plt.show()