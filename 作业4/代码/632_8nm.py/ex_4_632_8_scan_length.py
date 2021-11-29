import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
from scipy.signal import chirp, find_peaks, peak_widths

##########################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# “传说中，在扫描长度达到一定的时候，
#  有叠加噪声的波形分辨率会固定，然而本程序目前无法证实这一结论”
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 632.8nm
#   - 干涉图采样点数选取 2^16
#   - 叠加正弦噪声
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

def FindFWHM(Y0):
    peaks, _ = find_peaks(Y0)
    results_half = peak_widths(Y0, peaks, rel_height=0.5)
    print(results_half[0])
    
    results_full = peak_widths(Y0, peaks, rel_height=1)
    print(results_full[0])

    print(max(results_half[0]))
    return results_half

# 波长为632.8nm
laimda0 = 632.8*10**(-9)
# 79.1nm的采样间隔 
i = 79.1*10**(-9)
# 中心点的采样频
sigma0 = 1/laimda0
p1 = (-1)*(2**21)*79.1*10**(-9)
# 2**n个点 
p2 = (2**21-1)*79.1*10**(-9)
# 无补零 
x0 = np.arange(p1, p2, i)
# print("x0 length = %d" %len(x0))
n0 = n1 = 2**(int(np.log2(len(x0)))+1)
# n0 = len(x0)
print("n0 length = %d" %n0)

######################################################
#  此部分为叠加正弦噪声
######################################################
noise_sin1 = laimda0/10*np.sin(2*np.pi*(10**4)*x0)

I0 = np.cos(2*np.pi*sigma0*x0)
Iw0 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1))

Y0, Yw0 = GetFFT(I0, Iw0, n0)

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Y0.size, 1))

figure(1)
plt.plot(x0, noise_sin1)
plt.suptitle("Waveform of sinusoidal noise", fontsize = 25)

figure(2)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_0, Y0, marker='o', ms=5)
plt.plot(fs_0, best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3109 $m^{-1}$")
plt.xlim(1.578*(10**6), 1.582*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing sine noise")

subplot(2,2,4)
plt.plot(fs_0, Yw0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("Spectral graph of $I_0$ after superimposing sine noise FWHM = 3109 $m^{-1}$")
# plt.xlim(1.578*(10**6), 1.582*(10**6))
plt.xlim(1.52*(10**6), 1.65*(10**6))
# plt.xlim(1.580117*(10**6), 1.580473*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding sinusoidal noise", fontsize = 20)

plt.show()
