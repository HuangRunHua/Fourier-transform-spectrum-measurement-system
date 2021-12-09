import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *

##########################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 632.8nm
#   - 干涉图采样点数选取 2^12
#   - 叠加三种不同的噪声，分别为正弦噪声、随机噪声和线性噪声
##########################################################################

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
noise_sin1 = laimda0/16*np.sin(2*np.pi*(3*10**4)*x0)

I0 = np.cos(2*np.pi*sigma0*x0)
Iw0 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1))

Y0 = 2*abs(fft(I0,n0))
Y0_max = max(Y0);
Y0 = Y0/Y0_max
Y0 = Y0[:int(n0/2)]

Yw0 = 2*abs(fft(Iw0,n0))
Yw0_max = max(Yw0);
Yw0 = Yw0/Yw0_max
Yw0 = Yw0[:int(n0/2)]

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
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3109 $m^{-1}$")
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing sine noise")

subplot(2,2,4)
plt.plot(fs_0, Yw0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("Spectral graph of $I_0$ after superimposing sine noise FWHM = 3109 $m^{-1}$")
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding sinusoidal noise", fontsize = 20)

#########################################################
#  此部分为叠加随机噪声
#########################################################
noise_rand = laimda0/5*(rand(1,len(x0)))
Iw1 = np.cos(2*np.pi*sigma0*(x0 + noise_rand[0]))

Yw1 = 2*abs(fft(Iw1,n0))
Yw1_max=max(Yw1);
Yw1 = Yw1/Yw1_max
Yw1 = Yw1[:int(n0/2)]

# 设置频谱图的横坐标
fs_1 = 1/i*np.arange(n0/2)/n0

figure(3)
plt.plot(x0, noise_rand[0])
plt.suptitle("Waveform of random noise", fontsize = 25)

figure(4)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_1, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3109 $m^{-1}$")
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw1)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing random noise")

subplot(2,2,4)
plt.plot(fs_1, Yw1, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("Spectral graph of $I_0$ after superimposing random noise FWHM = 3103 $m^{-1}$")
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding random noise", fontsize = 20)

###################################################
#  此部分为叠加线性噪声
###################################################
noise_linear = linspace(0,i,len(x0))
Iw2 = np.cos(2*np.pi*sigma0*(x0 + noise_linear))

Yw2 = 2*abs(fft(Iw2,n0))
Yw2_max=max(Yw2);
Yw2 = Yw2/Yw2_max
Yw2 = Yw2[:int(n0/2)]

# 设置频谱图的横坐标
fs_2 = 1/i*np.arange(n0/2)/n0

figure(5)
plt.plot(x0, noise_linear)
plt.suptitle("Waveform of linear noise", fontsize = 25)

figure(6)
subplot(2,2,1)
plt.plot(x0, I0)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("$I_0$ Original Interferogram")


subplot(2,2,2)
plt.plot(fs_1, Y0, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve FWHM = 3109 $m^{-1}$")
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))

subplot(2,2,3)
plt.plot(x0, Iw2)
plt.xlim(-6*(10**(-6)), 6*(10**(-6)))
plt.title("Interferogram of $I_0$ after superimposing linear noise")

subplot(2,2,4)
plt.plot(fs_2, Yw2, marker='o', ms=5)
plt.plot(fs_0,best_Y0_average_range)
plt.title("Spectral graph of $I_0$ after superimposing linear noise FWHM = 3300 $m^{-1}$")
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))

plt.suptitle("632.8nm - Comparison of the original signal and the signal after adding linear noise", fontsize = 20)


############################################################
# 将所有干涉图和光谱曲线图放在一张图上
# 顺序如下：
#   - 原始波形
#   - 叠加正弦噪声后的波形
#   - 叠加随机噪声后的波形
#   - 叠加线性噪声后的波形
#############################################################
figure(7)
l1, = plt.plot(x0, I0)
l2, = plt.plot(x0, Iw0)
l3, = plt.plot(x0, Iw1)
l4, = plt.plot(x0, Iw2)
plt.legend(handles=[l1, l2, l3, l4],labels=['Original spectrum curve', 'Sine noise', 'Random noise', 'Linear noise'], loc='upper right')
plt.suptitle("632.8nm - Comparison between the original waveform and \n the waveform after adding three kinds of noise", fontsize = 20)
plt.xlim(-2*(10**(-6)), 2*(10**(-6)))
plt.xlabel("Scan Length$/m$")
plt.ylabel("Interference Intensity")

figure(8)
s1, = plt.plot(fs_0, Y0, marker='o', ms=5)
s2, = plt.plot(fs_0, Yw0, marker='p', ms=5)
s3, = plt.plot(fs_1, Yw1, marker='^', ms=5)
s4, = plt.plot(fs_2, Yw2, marker='s', ms=5)
s5, = plt.plot(fs_0,best_Y0_average_range, linewidth = 2.5)
plt.legend(handles=[s1, s2, s3, s4, s5],labels=['Original spectrum curve', 'Sine noise', 'Random noise', 'Linear noise', 'FWHM'], loc='upper right')
plt.suptitle("632.8nm - Comparison between the original waveform and the spectrum measurement curve \n after adding three kinds of noise", fontsize = 20)
#plt.xlim(1.52*(10**6), 1.65*(10**6))
plt.xlim(1.5758*(10**6), 1.58622*(10**6))
plt.xlabel("Wave number $/m^{-1}$")
plt.ylabel("Amplitude")

plt.show()
