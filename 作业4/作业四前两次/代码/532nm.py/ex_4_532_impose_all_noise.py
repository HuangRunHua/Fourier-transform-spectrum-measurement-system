import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
from scipy import signal
import random
import matplotlib.gridspec as gridspec

##########################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 532nm
#   - 干涉图采样点数选取 2^15（该点数下仿真图像最佳）
#   - 本实验叠加三种噪声---正弦噪声、随机噪声和线性噪声
#   - 实验目的在于模拟不同噪声同时叠加对信号的影响
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

t = linspace(0,i,len(x0))

# 第一种情况（作为参考情况）
noise_sin = laimda0/16*np.sin(2*np.pi*(1*10**4)*x0)
noise_rand = [random.uniform(-laimda0/5, laimda0/5) for _ in range(len(x0))]
noise_linear = i*signal.sawtooth(2 * np.pi * 8*10**7 * t, 0.5)

# 第二种情况（改变线性噪声的形状）
noise_sin1 = laimda0/16*np.sin(2*np.pi*(1*10**4)*x0)
noise_rand1 = [random.uniform(-laimda0/5, laimda0/5) for _ in range(len(x0))]
noise_linear1 = linspace(0,i,len(x0))

# 第三种情况（改变随机噪声的形状）
noise_sin2 = laimda0/16*np.sin(2*np.pi*(1*10**4)*x0)
noise_rand2 = [random.uniform(0, laimda0/5) for _ in range(len(x0))]
noise_linear2 = i*signal.sawtooth(2 * np.pi * 8*10**7 * t, 0.5)

# 第四种情况（改变正弦噪声的形状）
noise_sin3 = laimda0/16*np.sin(2*np.pi*(0.6*10**4)*x0)
noise_rand3 = [random.uniform(-laimda0/5, laimda0/5) for _ in range(len(x0))]
noise_linear3 = i*signal.sawtooth(2 * np.pi * 8*10**7 * t, 0.5)

I0 = np.cos(2*np.pi*sigma0*x0)
Iw0 = np.cos(2*np.pi*sigma0*(x0 + noise_sin + noise_rand + noise_linear))
Iw1 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1 + noise_rand1 + noise_linear1))
Iw2 = np.cos(2*np.pi*sigma0*(x0 + noise_sin2 + noise_rand2 + noise_linear2))
Iw3 = np.cos(2*np.pi*sigma0*(x0 + noise_sin3 + noise_rand3 + noise_linear3))

Y0, Yw0 = GetFFT(I0, Iw0, n0)
Y0, Yw1 = GetFFT(I0, Iw1, n0)
Y0, Yw2 = GetFFT(I0, Iw2, n0) 
Y0, Yw3 = GetFFT(I0, Iw3, n0) 

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Y0.size, 1))

#########################################################
# 第一种情况结果
#########################################################
figure(1)

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
plt.plot(x0, noise_sin)
plt.xlim(-0.00006, 0.00006)
plt.title("Waveform of sinusoidal noise")

subplot(gs[0, 2:4])
plt.plot(x0, noise_rand)
plt.xlim(-0.00005, 0.00005)
plt.title("Waveform of random noise")

subplot(gs[0, 4:6])
linear1, = plt.plot(x0, noise_linear)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:6])
Yw0_pic0, =plt.plot(fs_0, Y0, marker='^', ms=5)
Yw0_pic1, = plt.plot(fs_0, Yw0, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic0, Yw0_pic1, bf],labels=['Spectral graph(No Noise) FWHM = 385.8 $m^{-1}$', 'Spectral graph(Imposing All Noise) FWHM = 393.9 $m^{-1}$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing all noise")
plt.xlim(1.866*(10**6), 1.8939*(10**6))

plt.suptitle("532nm - Comparison of the Original Signal and the Signal after Adding All Noise", fontsize = 20)

#########################################################
# 第二种情况结果
#########################################################
figure(2)

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
plt.plot(x0, noise_sin1)
plt.xlim(-0.00006, 0.00006)
plt.title("Waveform of sinusoidal noise")

subplot(gs[0, 2:4])
plt.plot(x0, noise_rand1)
plt.xlim(-0.00005, 0.00005)
plt.title("Waveform of random noise")

subplot(gs[0, 4:6])
linear1, = plt.plot(x0, noise_linear1)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:6])
Yw0_pic0, =plt.plot(fs_0, Y0, marker='^', ms=5)
Yw0_pic2, = plt.plot(fs_0, Yw1, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic0, Yw0_pic2, bf],labels=['Spectral graph(No Noise) FWHM = 385.8 $m^{-1}$', 'Spectral graph(Imposing All Noise) FWHM = 442.3 $m^{-1}$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing all noise")
plt.xlim(1.866*(10**6), 1.8939*(10**6))

plt.suptitle("532nm - Comparison of the Original Signal and the Signal after Adding All Noise", fontsize = 20)

#########################################################
# 第三种情况结果
#########################################################
figure(3)

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
plt.plot(x0, noise_sin2)
plt.xlim(-0.00006, 0.00006)
plt.title("Waveform of sinusoidal noise")

subplot(gs[0, 2:4])
plt.plot(x0, noise_rand2)
plt.xlim(-0.00005, 0.00005)
plt.title("Waveform of random noise")

subplot(gs[0, 4:6])
linear1, = plt.plot(x0, noise_linear2)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:6])
Yw0_pic0, =plt.plot(fs_0, Y0, marker='^', ms=5)
Yw0_pic3, = plt.plot(fs_0, Yw2, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic0, Yw0_pic3, bf],labels=['Spectral graph(No Noise) FWHM = 385.8 $m^{-1}$', 'Spectral graph(Imposing All Noise) FWHM = 391.5 $m^{-1}$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing all noise")
plt.xlim(1.866*(10**6), 1.8939*(10**6))

plt.suptitle("532nm - Comparison of the Original Signal and the Signal after Adding All Noise", fontsize = 20)

#########################################################
# 第四种情况结果
#########################################################
figure(4)

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
plt.plot(x0, noise_sin3)
plt.xlim(-0.00006, 0.00006)
plt.title("Waveform of sinusoidal noise")

subplot(gs[0, 2:4])
plt.plot(x0, noise_rand3)
plt.xlim(-0.00005, 0.00005)
plt.title("Waveform of random noise")

subplot(gs[0, 4:6])
linear1, = plt.plot(x0, noise_linear3)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:6])
Yw0_pic0, =plt.plot(fs_0, Y0, marker='^', ms=5)
Yw0_pic4, = plt.plot(fs_0, Yw3, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic0, Yw0_pic4, bf],labels=['Spectral graph(No Noise) FWHM = 385.8 $m^{-1}$', 'Spectral graph(Imposing All Noise) FWHM = 392.3 $m^{-1}$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing all noise")
plt.xlim(1.866*(10**6), 1.8939*(10**6))

plt.suptitle("532nm - Comparison of the Original Signal and the Signal after Adding All Noise", fontsize = 20)


plt.show()
