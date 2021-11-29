import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *

##########################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 632.8nm
#   - 干涉图采样点数选取 2^12（该点数下仿真图像最佳）
#   - 本实验只叠加一种噪声---正弦噪声
#   - 实验目的在于模拟不同周期的正弦噪声对信号的影响
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

######################################################
#  此部分为叠加正弦噪声
######################################################
noise_sin1 = laimda0/16*np.sin(2*np.pi*(0.6*10**4)*x0)
noise_sin2 = laimda0/16*np.sin(2*np.pi*(1*10**4)*x0)
noise_sin4 = laimda0/16*np.sin(2*np.pi*(2.5*10**4)*x0)
noise_sin5 = laimda0/16*np.sin(2*np.pi*(4*10**4)*x0)

I0 = np.cos(2*np.pi*sigma0*x0)
Iw0 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1))
Iw0_1 = np.cos(2*np.pi*sigma0*(x0 + noise_sin2))
Iw0_3 = np.cos(2*np.pi*sigma0*(x0 + noise_sin4))
Iw0_4 = np.cos(2*np.pi*sigma0*(x0 + noise_sin5))

Y0, Yw0 = GetFFT(I0, Iw0, n0)
Y0, Yw0_1 = GetFFT(I0, Iw0_1, n0)
Y0, Yw0_3 = GetFFT(I0, Iw0_3, n0)
Y0, Yw0_4 = GetFFT(I0, Iw0_4, n0)

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Y0.size, 1))

figure(1)
sin1, = plt.plot(x0, noise_sin1)
sin2, = plt.plot(x0, noise_sin2)
sin4, = plt.plot(x0, noise_sin4)
sin5, = plt.plot(x0, noise_sin5)
plt.xlim(-0.00006, 0.00006)
plt.legend(handles=[sin1, sin2, sin4, sin5],labels=['noise_sin1', 'noise_sin2', 'noise_sin4', 'noise_sin5'], loc='upper right')
plt.suptitle("Waveform of sinusoidal noise", fontsize = 25)

figure(2)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_0, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing sine noise")

subplot(2,2,4)
Yw0_pic1, = plt.plot(fs_0, Yw0, marker='o', ms=5)
bf, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic1, bf],labels=['$\omega=0.6x10^4 rad/s$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing sine noise FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding sinusoidal noise", fontsize = 20)

figure(3)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_0, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))
#plt.xlim(1.53095*(10**6), 1.62921*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw0_1)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing sine noise")

subplot(2,2,4)
Yw0_pic2, = plt.plot(fs_0, Yw0_1, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic2, bf],labels=['$\omega=1x10^4 rad/s$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing sine noise FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))
#plt.xlim(1.53095*(10**6), 1.62921*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding sinusoidal noise", fontsize = 20)

figure(5)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_0, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))
#plt.xlim(1.53095*(10**6), 1.62921*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw0_3)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing sine noise")

subplot(2,2,4)
Yw0_pic3, = plt.plot(fs_0, Yw0_3, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic3, bf],labels=['$\omega=2.5x10^4 rad/s$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing sine noise FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))
#plt.xlim(1.53095*(10**6), 1.62921*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding sinusoidal noise", fontsize = 20)

figure(6)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_0, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))
#plt.xlim(1.53095*(10**6), 1.62921*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw0_4)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing sine noise")

subplot(2,2,4)
Yw0_pic4, = plt.plot(fs_0, Yw0_4, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[Yw0_pic4, bf],labels=['$\omega=4x10^4 rad/s$', 'FWHM'], loc='upper right')
plt.title("Spectral graph of $I_0$ after superimposing sine noise FWHM = 3086 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.67*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding sinusoidal noise", fontsize = 20)

############################################################################################
# 此部分将所有光谱曲线放在一张图内
# 按频率逐渐增大的方式排列
############################################################################################
figure(7)
sine1, = plt.plot(fs_0, Y0, marker='o', ms=5)
sine2, = plt.plot(fs_0, Yw0, marker='*', ms=5)
sine3, = plt.plot(fs_0, Yw0_1, marker='p', ms=5)
sine5, = plt.plot(fs_0, Yw0_3, marker='s', ms=5)
sine6, = plt.plot(fs_0, Yw0_4, marker='<', ms=5)
sine7, = plt.plot(fs_0,best_Y0_average_range)
plt.legend(handles=[sine1, sine2, sine3, sine5, sine6, sine7],labels=['Origneal Spectrum Curve', '$\omega=0.6x10^4 rad/s$', '$\omega=1x10^4 rad/s$', '$\omega=2.5x10^4 rad/s$', '$\omega=4x10^4 rad/s$', 'FWHM'], loc='upper right')
plt.xlim(1.50*(10**6), 1.67*(10**6))
# #plt.xlim(1.53095*(10**6), 1.62921*(10**6))

plt.suptitle("632.8nm - Comparison of the Original Signal And the Signal after \n Adding Sinusoidal Noise in One Picture", fontsize = 20)

plt.show()
