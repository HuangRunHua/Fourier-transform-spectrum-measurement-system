import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
import random

##########################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 532nm
#   - 干涉图采样点数选取 2^15
#   - 本程序只叠加一种噪声----随机噪声
##########################################################################

def GetFFT(I0, Iw0, n0):
    Y0 = 2*abs(fft(I0,n0))
    Y0_max = max(Y0);
    Y0 = Y0/Y0_max
    Y0 = Y0[:int(n0/2)]

    Yw0 = 2*abs(fft(Iw0,n0))
    Yw0_max = max(Yw0);
    Yw0 = Yw0/Yw0_max
    Yw0 = Yw0[:int(n0/2)]
    return Y0, Yw0

# 波长为532nm
laimda0 = 532*10**(-9)
# 79.1nm的采样间隔 
i = 79.1*10**(-9)
# 中心点的采样频
sigma0 = 1/laimda0
p1 = (-1)*(2**14)*79.1*10**(-9)
# 2**n个点 
p2 = (2**14-1)*79.1*10**(-9)
# 无补零 
x0 = np.arange(p1, p2, i)
n0 = n1 = 2**(int(np.log2(len(x0)))+1)
print("n0 length = %d" %n0)

I0 = np.cos(2*np.pi*sigma0*x0)

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0

#########################################################
#  此部分为叠加随机噪声
#########################################################
noise_rand = [random.uniform(-laimda0/5, laimda0/5) for _ in range(len(x0))]
# noise_rand = [random.uniform(0, laimda0/5) for _ in range(len(x0))]

Iw1 = np.cos(2*np.pi*sigma0*(x0 + noise_rand))

Y0, Yw1 = GetFFT(I0, Iw1, n0)

# 设置频谱图的横坐标
fs_1 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Yw1.size, 1))

figure(1)
plt.plot(x0, noise_rand)
plt.xlim(-0.00005, 0.00005)
plt.suptitle("Waveform of random noise", fontsize = 25)

figure(2)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_1, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw1)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing random noise")

subplot(2,2,4)
plt.plot(fs_1, Yw1, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("After superimposing random noise FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding random noise", fontsize = 20)

plt.show()
