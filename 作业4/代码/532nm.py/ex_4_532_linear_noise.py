import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
from scipy import signal

##########################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 532nm
#   - 干涉图采样点数选取 2^15
#   - 本实验只叠加一种噪声----线性噪声
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

###################################################
#  此部分为叠加线性噪声
###################################################
I0 = np.cos(2*np.pi*sigma0*x0)

noise_linear = linspace(0,i,len(x0))
sig = np.sin(2 * np.pi * 8 *10**7 * noise_linear)
# 以下三个为等腰三角波
noise_linear1 = i*signal.sawtooth(2 * np.pi * 4*10**7 * noise_linear, 0.5)
noise_linear2 = i*signal.sawtooth(2 * np.pi * 8*10**7 * noise_linear, 0.5)
noise_linear3 = i*signal.sawtooth(2 * np.pi * 10*10**7 * noise_linear, 0.5)
# 以下两个为矩形波
noise_linear4 = i*signal.square(2 * np.pi * 10*10**7 * noise_linear, duty=(noise_linear + 1)/2)
noise_linear5 = i*signal.square(2 * np.pi * 10*10**7 * noise_linear, duty=(sig + 1)/2)
noise_linear6 = i*signal.square(2 * np.pi * 20*10**7 * noise_linear, duty=(noise_linear + 1)/2)
# 锯齿波
noise_linear7 = i*signal.sawtooth(2 * np.pi * 4*10**7 * noise_linear)
noise_linear8 = i*signal.sawtooth(2 * np.pi * 8*10**7 * noise_linear)

# 直线
Iw2 = np.cos(2*np.pi*sigma0*(x0 + noise_linear))
# 三角波
Iw3 = np.cos(2*np.pi*sigma0*(x0 + noise_linear1))
Iw4 = np.cos(2*np.pi*sigma0*(x0 + noise_linear2))
Iw5 = np.cos(2*np.pi*sigma0*(x0 + noise_linear3))
# 矩形波
Iw6 = np.cos(2*np.pi*sigma0*(x0 + noise_linear4))
Iw7 = np.cos(2*np.pi*sigma0*(x0 + noise_linear5))
Iw8 = np.cos(2*np.pi*sigma0*(x0 + noise_linear6))
# 锯齿波
Iw9 = np.cos(2*np.pi*sigma0*(x0 + noise_linear7))
Iw10 = np.cos(2*np.pi*sigma0*(x0 + noise_linear8))

# 直线
Y0, Yw2 = GetFFT(I0, Iw2, n0)
# 三角波
Y0, Yw3 = GetFFT(I0, Iw3, n0)
Y0, Yw4 = GetFFT(I0, Iw4, n0)
Y0, Yw5 = GetFFT(I0, Iw5, n0)
# 矩形波
Y0, Yw6 = GetFFT(I0, Iw6, n0)
Y0, Yw7 = GetFFT(I0, Iw7, n0)
Y0, Yw8 = GetFFT(I0, Iw8, n0)
# 锯齿波
Y0, Yw9 = GetFFT(I0, Iw9, n0)
Y0, Yw10 = GetFFT(I0, Iw10, n0)

# 设置频谱图的横坐标
fs_2 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Yw2.size, 1))

# 绘制所有原始线性噪声图
figure(1)
subplot(3,3,1)
linear1, = plt.plot(x0, noise_linear)
subplot(3,3,2)
linear8, = plt.plot(x0, noise_linear7)
subplot(3,3,3)
linear9, = plt.plot(x0, noise_linear8)
subplot(3,3,4)
linear2, = plt.plot(x0, noise_linear1)
subplot(3,3,5)
linear3, = plt.plot(x0, noise_linear2)
subplot(3,3,6)
linear4, = plt.plot(x0, noise_linear3)
subplot(3,3,7)
linear5, = plt.plot(x0, noise_linear4)
subplot(3,3,8)
linear6, = plt.plot(x0, noise_linear5)
subplot(3,3,9)
linear7, = plt.plot(x0, noise_linear6)
plt.suptitle("Waveform of linear noise", fontsize = 25)


# 绘制直线
figure(2)
#######################################################
# 每张Figure添加噪声图片：
#   - linear_i, = plt.plot(x0, noise_linear_j) 
#     其中i = 1,...,9; j = 0,...,8
# 注意修改suplot的索引
# 添加的噪声顺序按索引从小到大排列
#######################################################
subplot(3,1,1)
linear1, = plt.plot(x0, noise_linear)
plt.title("Linear Noise 1")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw2)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw2, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 400 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

########################################################
# 从第二张Figure开始每张Figure添加此语句
########################################################
plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 锯齿波 1
figure(3)
subplot(3,1,1)
linear8, = plt.plot(x0, noise_linear7)
plt.title("Linear Noise 2")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw9)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw9, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 392 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 锯齿波 2
figure(4)
subplot(3,1,1)
linear9, = plt.plot(x0, noise_linear8)
plt.title("Linear Noise 3")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw10)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw10, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 410 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 三角波 1
figure(5)
subplot(3,1,1)
linear2, = plt.plot(x0, noise_linear1)
plt.title("Linear Noise 4")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw3)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw3, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 399 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 三角波 2
figure(6)
subplot(3,1,1)
linear3, = plt.plot(x0, noise_linear2)
plt.title("Linear Noise 5")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw4)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw4, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 390 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 三角波 3
figure(7)
subplot(3,1,1)
linear4, = plt.plot(x0, noise_linear3)
plt.title("Linear Noise 6")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")

subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw5)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw5, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 矩形波 1
figure(8)
subplot(3,1,1)
linear5, = plt.plot(x0, noise_linear4)
plt.title("Linear Noise 7")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw6)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw6, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 395 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))
# plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 矩形波 2
figure(9)
subplot(3,1,1)
linear6, = plt.plot(x0, noise_linear5)
plt.title("Linear Noise 8")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw7)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw7, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 389 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

# 矩形波 3
figure(10)
subplot(3,1,1)
linear7, = plt.plot(x0, noise_linear6)
plt.title("Linear Noise 9")

subplot(3,2,3)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(3,2,4)
plt.plot(fs_2, Y0, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 385 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

subplot(3,2,5)
plt.plot(x0, Iw8)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(3,2,6)
plt.plot(fs_2, Yw8, marker='o', ms=5)
plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise FWHM = 389 $m^{-1}$")
plt.xlim(1.87*(10**6), 1.89*(10**6))

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.suptitle("532nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)

plt.show()
