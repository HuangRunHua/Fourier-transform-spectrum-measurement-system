import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
import matplotlib.gridspec as gridspec
import random

##########################################################################
# 本程序模拟动镜倾斜误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 632.8nm
#   - 干涉图采样点数选取 2^12（该点数下仿真图像最佳）
#   - 本实验只叠加两种正弦噪声噪声
#   - 实验目的在于模拟不同频率的正弦噪声对信号的影响
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

# 波长为632.8nm
laimda0 = 632.8*10**(-9)
# 79.1nm的采样间隔 
i = 79.1*10**(-9)
# 中心点的采样频
sigma0 = 1/laimda0
p1 = (-1)*(2**11)*79.1*10**(-9)
# 2**n个点 
p2 = (2**11-1)*79.1*10**(-9)
# 无补零 
x0 = np.arange(p1, p2, i)
n0 = n1 = 2**(int(np.log2(len(x0)))+1)
print("n0 length = %d" %n0)


noise1 = [random.uniform(-1, 1) for _ in range(len(x0))]
# noise1 = np.sin(2*np.pi*sigma0/8*x0)
# noise1 = np.sin(2*np.pi*sigma0/4*x0)
# noise1 = np.sin(2*np.pi*sigma0/2*x0)
noise2 = laimda0/16*np.sin(2*np.pi*(0.4*10**4)*x0)

I0 = np.cos(2*np.pi*sigma0*x0)
I00 = np.cos(2*np.pi*sigma0*(x0 + noise2))
# Iw0 = I0 + noise1
Iw0 = I00 + noise1

Y0, Yw0 = GetFFT(I0, Iw0, n0)

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Y0.size, 1))
best_Y1_average_range = 0.4745*np.ones((Y0.size, 1))

gs = gridspec.GridSpec(4, 8)
gs.update(wspace=0.5, hspace=0.7)



figure(1)
subplot(gs[0, 0:4])
plt.plot(x0, noise1)
# plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Error of Moving Mirror Tilt")


subplot(gs[0, 4:8])
plt.plot(x0, noise2)
plt.title("Error of Sampling Interval")



subplot(gs[1, 0:4])
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(gs[1, 4:8])
Y0_pic1, = plt.plot(fs_0, Y0, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Y0_pic1, bf],labels=['Original wave', 'FWHM = 3086 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.50*(10**6), 1.67*(10**6))
plt.xlabel('Wave number($m^{-1}$)')

subplot(gs[2, 0:4])
plt.plot(x0, Iw0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing noise")

subplot(gs[2, 4:8])
Yw0_pic1, = plt.plot(fs_0, Yw0, marker='o', ms=5)
# bf, = plt.plot(fs_0,best_Y0_average_range)
bf, = plt.plot(fs_0,best_Y1_average_range)
# plt.legend(handles=[Yw0_pic1, bf],labels=[r'After imposing noise', 'FWHM = 3086 $m^{-1}$'], loc='upper right')
plt.legend(handles=[Yw0_pic1, bf],labels=[r'After imposing noise', 'FWHM = 3435 $m^{-1}$'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing noise")
plt.xlim(1.50*(10**6), 1.67*(10**6))
# plt.xlim(0.07098*(10**6), 0.1235*(10**6))
plt.xlabel('Wave number($m^{-1}$)')


subplot(gs[3, :8])

Yw0_pic1, = plt.plot(fs_0, Yw0, marker='o', ms=5)
Y0_pic1, = plt.plot(fs_0, Y0, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[ Yw0_pic1, Y0_pic1, bf],labels=[r'After imposing noise', 'Original wave'], bbox_to_anchor=(0.85,0.45), loc='best')
plt.xlim(0*(10**6), 1.7*(10**6))
plt.xlabel('Wave number($m^{-1}$)')
plt.title("Spectral graph of $I_0$ before and after superimposing noise")

plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Noise", fontsize = 20)


plt.show()
