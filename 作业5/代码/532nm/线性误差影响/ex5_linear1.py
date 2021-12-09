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
#   - 干涉图采样点数选取 2^15
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


Iw1 = I0 + noise_linear1
Iw2 = I0 + noise_linear2
Iw3 = I0 + noise_linear3
Iw4 = I0 + noise_linear4
Iw5 = I0 + noise_linear5
Iw6 = I0 + noise_linear6
Iw7 = I0 + noise_linear7
Iw8 = I0 + noise_linear8
Iw9 = I0 + noise_linear9

Y0, Yw1 = GetFFT(I0, Iw1, n0)
Y0, Yw2 = GetFFT(I0, Iw2, n0)
Y0, Yw3 = GetFFT(I0, Iw3, n0)
Y0, Yw4 = GetFFT(I0, Iw4, n0)
Y0, Yw5 = GetFFT(I0, Iw5, n0)
Y0, Yw6 = GetFFT(I0, Iw6, n0)
Y0, Yw7 = GetFFT(I0, Iw7, n0)
Y0, Yw8 = GetFFT(I0, Iw8, n0)
Y0, Yw9 = GetFFT(I0, Iw9, n0)

# 设置频谱图的横坐标
fs_1 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Yw1.size, 1))

####################################################################
# 线性噪声 1
####################################################################
figure(1)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear1)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
# plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw1)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw1, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.9 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 2
####################################################################
figure(2)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear2)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw2)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw2, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.9 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 3
####################################################################
figure(3)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear3)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw3)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw3, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.9 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 4
####################################################################
figure(4)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear4)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw4)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw4, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.9 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 5
####################################################################
best_Y0_average_range5 = 0.465*np.ones((Yw1.size, 1))

figure(5)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear5)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw5)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw5, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range5)
plt.legend(handles=[p2],labels=['FWHM = 394.1 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)


####################################################################
# 线性噪声 6
####################################################################
best_Y0_average_range6 = 0.4245*np.ones((Yw1.size, 1))
figure(6)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear6)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw6)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw6, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range6)
plt.legend(handles=[p2],labels=['FWHM = 395.8 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 7
####################################################################
best_Y0_average_range7 = 0.4275*np.ones((Yw1.size, 1))
figure(7)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear7)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw7)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw7, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.8 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 8
####################################################################
best_Y0_average_range8 = 0.4275*np.ones((Yw1.size, 1))
figure(8)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear8)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw8)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw8, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.8 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

####################################################################
# 线性噪声 9
####################################################################
best_Y0_average_range9 = 0.4275*np.ones((Yw1.size, 1))
figure(9)
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :6])
plt.plot(x0, noise_linear9)
plt.title("Waveform of linear noise")

subplot(gs[1, 0:3])
s1, = plt.plot(x0, I0)
plt.legend(handles=[s1],labels=['Sample number = $2^{15}$'], loc='upper right')
plt.title("$I_0$ Original Interferogram")

subplot(gs[1, 3:6])
s1, = plt.plot(fs_1, Y0, marker='o', ms=5)
p1, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p1],labels=['FWHM = 385.8 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(gs[2, 0:3])
plt.plot(x0, Iw9)
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(gs[2, 3:6])
plt.plot(fs_1, Yw9, marker='o', ms=5)
p2, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[p2],labels=['FWHM = 386.8 $m^{-1}$'], loc='upper right')
plt.title("After superimposing linear noise")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)


plt.show()
