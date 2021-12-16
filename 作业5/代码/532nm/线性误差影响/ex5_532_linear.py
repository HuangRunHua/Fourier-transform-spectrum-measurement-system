import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
import matplotlib.gridspec as gridspec
from scipy import signal

##########################################################################
# 本程序模拟动镜倾斜误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 532nm
#   - 干涉图采样点数选取 2^12
#   - 本程序只叠加一种噪声----线性噪声
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
p1 = (-1)*(2**11)*79.1*10**(-9)
# 2**n个点 
p2 = (2**11-1)*79.1*10**(-9)
# 无补零 
x0 = np.arange(p1, p2, i)
n0 = n1 = 2**(int(np.log2(len(x0)))+1)
print("n0 length = %d" %n0)

I0 = np.cos(2*np.pi*sigma0*x0)

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0


noise_linear1 = linspace(0,1,len(x0))
sig = np.sin(2 * np.pi * 10 * noise_linear1)
# 以下三个为等腰三角波
noise_linear2 = signal.sawtooth(2 * np.pi * 10 * noise_linear1, 0.5)
noise_linear3 = signal.sawtooth(2 * np.pi * 15 * noise_linear1, 0.5)
noise_linear4 = signal.sawtooth(2 * np.pi * 20 * noise_linear1, 0.5)
# 以下两个为矩形波
noise_linear5 = signal.square(2 * np.pi * 10 * noise_linear1, duty=(noise_linear1 + 1)/2)
noise_linear6 = signal.square(2 * np.pi * 15 * noise_linear1, duty=(sig + 1)/2)
noise_linear7 = signal.square(2 * np.pi * 20 * noise_linear1, duty=(noise_linear1 + 1)/2)
# 锯齿波
noise_linear8 = signal.sawtooth(2 * np.pi * 4 * noise_linear1)
noise_linear9 = signal.sawtooth(2 * np.pi * 8 * noise_linear1)


Iw1 = I0 + noise_linear9

Y0, Yw1 = GetFFT(I0, Iw1, n0)

# 设置频谱图的横坐标
fs_1 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Yw1.size, 1))

####################################################################
# 线性噪声 1
####################################################################
figure(1)
# gs = gridspec.GridSpec(4, 8)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{12}$'], loc='upper right')
# plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")

subplot(gs[0, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 3086 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.82*(10**6), 1.94*(10**6))

subplot(gs[1, 0:3])
plt.plot(x0, Iw1)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

best_Y1_average_range = 0.4275*np.ones((Yw1.size, 1))

subplot(gs[1, 3:6])
plt.plot(fs_1, Yw1, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 3087 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.82*(10**6), 1.94*(10**6))


subplot(gs[2, :6])
sn1, = plt.plot(fs_1, Yw1, marker='.', ms=5)
s1, = plt.plot(fs_1, Y0, marker='.', ms=5)
# p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[s1, sn1, p2],labels=[r'Original wave', 'Linear noise'], bbox_to_anchor=(0.9,1), loc='best')
# plt.legend(handles=[s1, sn1, p2],labels=[r'Original wave', 'Linear noise'], loc='upper right')
plt.xlabel('Wave number($m^{-1}$)')
# plt.xlim(-10000, 0.4*(10**6))
# plt.xlim(-10000, 1.74*(10**6))
# plt.xlim(-10000, 2*(10**6))
plt.xlim(-10000, 1.97*(10**6))
plt.title("Spectral graph of $I_0$ before and after superimposing linear noise")

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)


plt.show()
