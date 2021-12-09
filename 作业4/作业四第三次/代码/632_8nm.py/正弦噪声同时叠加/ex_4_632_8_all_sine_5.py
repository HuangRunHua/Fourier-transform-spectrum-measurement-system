import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
import matplotlib.gridspec as gridspec

#################################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 632.8nm
#   - 干涉图采样点数选取 2^15（该点数下仿真图像最佳）
#   - 本实验只叠加一种噪声---正弦噪声
#       - 频率分别为0.2*10^3, 0.21*10^3, 0.22*10^3, 0.23*10^3, 0.24*10^3 rad/s
#   - 实验目的在于模拟不同周期的正弦噪声同时叠加后对信号的影响
#   - 分别叠加2种、3种与4种噪声
#################################################################################

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
p1 = (-1)*(2**14)*79.1*10**(-9)
# 2**n个点 
p2 = (2**14-1)*79.1*10**(-9)
# 无补零 
x0 = np.arange(p1, p2, i)
n0 = n1 = 2**(int(np.log2(len(x0)))+1)
print("n0 length = %d" %n0)

######################################################
#  此部分为叠加正弦噪声
######################################################
noise_sin1 = laimda0/16*np.sin(2*np.pi*(0.20*10**3)*x0)
noise_sin2 = laimda0/16*np.sin(2*np.pi*(0.21*10**3)*x0)
noise_sin3 = laimda0/16*np.sin(2*np.pi*(0.22*10**3)*x0)
noise_sin4 = laimda0/16*np.sin(2*np.pi*(0.23*10**3)*x0)
noise_sin5 = laimda0/16*np.sin(2*np.pi*(0.24*10**3)*x0)

I0 = np.cos(2*np.pi*sigma0*x0)

I_sin_all_1 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1))
I_sin_all_2 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1 + noise_sin2))
I_sin_all_3 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1 + noise_sin2 + noise_sin3))
I_sin_all_4 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1 + noise_sin2 + noise_sin3 + noise_sin4))
I_sin_all_5 = np.cos(2*np.pi*sigma0*(x0 + noise_sin1 + noise_sin2 + noise_sin3 + noise_sin4 + noise_sin5))

Y0, Y_sine_all_1 = GetFFT(I0, I_sin_all_1, n0)
Y0, Y_sine_all_2 = GetFFT(I0, I_sin_all_2, n0)
Y0, Y_sine_all_3 = GetFFT(I0, I_sin_all_3, n0)
Y0, Y_sine_all_4 = GetFFT(I0, I_sin_all_4, n0)
Y0, Y_sine_all_5 = GetFFT(I0, I_sin_all_5, n0)

# 设置频谱图的横坐标
fs_0 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Y0.size, 1))

#########################################################
# 只叠加一种波
#########################################################
figure(1)
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :4])
sin1, = plt.plot(x0, noise_sin1)
plt.title("$\omega=0.2x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[1, 0:2])
plt.plot(fs_0, Y0, marker='o', ms=5)
FWHM_original, = plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve")
plt.legend(handles=[FWHM_original],labels=['$FWHM = 385.8 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

subplot(gs[1, 2:4])
plt.plot(fs_0, Y_sine_all_1, marker='o', ms=5)
FWHM_sine_1, = plt.plot(fs_0,best_Y0_average_range)
plt.title("After imposing sine noise")
plt.legend(handles=[FWHM_sine_1],labels=['$FWHM = 467.5 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Sinusoidal Noise", fontsize = 20)

#########################################################
# 同时叠加两种波
#########################################################
figure(2)
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
sin1, = plt.plot(x0, noise_sin1)
plt.title("$\omega=0.2x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 2:4])
sin2, = plt.plot(x0, noise_sin2)
plt.title("$\omega=0.21x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[1, 0:2])
plt.plot(fs_0, Y0, marker='o', ms=5)
FWHM_original, = plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve")
plt.legend(handles=[FWHM_original],labels=['$FWHM = 385.8 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

subplot(gs[1, 2:4])
plt.plot(fs_0, Y_sine_all_2, marker='o', ms=5)
FWHM_sine_2, = plt.plot(fs_0,best_Y0_average_range)
plt.title("After imposing sine noise")
plt.legend(handles=[FWHM_sine_2],labels=['$FWHM = 640.5 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Sinusoidal Noise", fontsize = 20)

#########################################################
# 同时叠加三种波
#########################################################
figure(3)
gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
sin1, = plt.plot(x0, noise_sin1)
plt.title("$\omega=0.2x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 2:4])
sin2, = plt.plot(x0, noise_sin2)
plt.title("$\omega=0.21x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 4:6])
sin3, = plt.plot(x0, noise_sin3)
plt.title("$\omega=0.22x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[1, 0:3])
plt.plot(fs_0, Y0, marker='o', ms=5)
FWHM_original, = plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve")
plt.legend(handles=[FWHM_original],labels=['$FWHM = 385.8 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

subplot(gs[1, 3:6])
plt.plot(fs_0, Y_sine_all_3, marker='o', ms=5)
FWHM_sine_3, = plt.plot(fs_0,best_Y0_average_range)
plt.title("After imposing sine noise")
plt.legend(handles=[FWHM_sine_3],labels=['$FWHM = 959.2 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Sinusoidal Noise", fontsize = 20)


#########################################################
# 同时叠加四种波
#########################################################
figure(4)
gs = gridspec.GridSpec(2, 8)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
sin1, = plt.plot(x0, noise_sin1)
plt.title("$\omega=0.2x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 2:4])
sin2, = plt.plot(x0, noise_sin2)
plt.title("$\omega=0.21x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 4:6])
sin3, = plt.plot(x0, noise_sin3)
plt.title("$\omega=0.22x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 6:8])
sin4, = plt.plot(x0, noise_sin4)
plt.title("$\omega=0.23x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[1, 0:4])
plt.plot(fs_0, Y0, marker='o', ms=5)
FWHM_original, = plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve")
plt.legend(handles=[FWHM_original],labels=['$FWHM = 385.8 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

subplot(gs[1, 4:8])
plt.plot(fs_0, Y_sine_all_4, marker='o', ms=5)
FWHM_sine_4, = plt.plot(fs_0,best_Y0_average_range)
plt.title("After imposing sine noise")
plt.legend(handles=[FWHM_sine_4],labels=['$FWHM = 690.4 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Sinusoidal Noise", fontsize = 20)

#########################################################
# 同时叠加五种波
#########################################################
figure(5)
gs = gridspec.GridSpec(2, 10)
gs.update(wspace=0.5, hspace=0.3)

subplot(gs[0, :2])
sin1, = plt.plot(x0, noise_sin1)
plt.title("$\omega=0.2x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 2:4])
sin2, = plt.plot(x0, noise_sin2)
plt.title("$\omega=0.21x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 4:6])
sin3, = plt.plot(x0, noise_sin3)
plt.title("$\omega=0.22x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 6:8])
sin4, = plt.plot(x0, noise_sin4)
plt.title("$\omega=0.23x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[0, 8:10])
sin4, = plt.plot(x0, noise_sin5)
plt.title("$\omega=0.24x10^3 rad/s$")
plt.xlim(-0.002, 0.002)

subplot(gs[1, 0:5])
plt.plot(fs_0, Y0, marker='o', ms=5)
FWHM_original, = plt.plot(fs_0,best_Y0_average_range)
plt.title("$I_0$ Original spectrum curve")
plt.legend(handles=[FWHM_original],labels=['$FWHM = 385.8 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

subplot(gs[1, 5:10])
plt.plot(fs_0, Y_sine_all_5, marker='o', ms=5)
FWHM_sine_5, = plt.plot(fs_0,best_Y0_average_range)
plt.title("After imposing sine noise")
plt.legend(handles=[FWHM_sine_5],labels=['$FWHM = 433.9 m^{-1}$'], loc='upper right')
plt.xlim(1.57*(10**6), 1.59*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Sinusoidal Noise", fontsize = 20)

####################################################################################
# 将所有波形放在一起
####################################################################################
figure(6)
pic, = plt.plot(fs_0, Y0, marker='o', ms=5)
pic1, = plt.plot(fs_0, Y_sine_all_1, marker='^', ms=5)
pic2, = plt.plot(fs_0, Y_sine_all_2, marker='*', ms=5)
pic3, = plt.plot(fs_0, Y_sine_all_3, marker='.', ms=5)
pic4, = plt.plot(fs_0, Y_sine_all_4, marker='x', ms=5)
pic5, = plt.plot(fs_0, Y_sine_all_5, marker='v', ms=5)
FWHM_sine_6, = plt.plot(fs_0,best_Y0_average_range, linewidth=3)
plt.legend(handles=[pic, pic1, pic2, pic3, pic4, pic5, FWHM_sine_6],labels=['Original', 'One Sine Noise', 'Two Sine Noise', 'Three Sine Noise', 'Four Sine Noise', 'Five Sine Noise', 'FWHM'], loc='upper right')
plt.xlim(1.578*(10**6), 1.5826*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')
plt.suptitle("632.8nm - Comparison of the Original Signal and the Signal after Adding Sinusoidal Noise", fontsize = 20)

plt.show()
