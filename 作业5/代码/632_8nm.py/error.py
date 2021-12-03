import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *

##########################################################################
# 本程序旨在绘制加噪声后分辨率与真实分辨率的误差比
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 632.8nm
#   - 干涉图采样点数选取 2^14～2^27
#   - 叠加正弦噪声：noise_sin_1 = laimda_0/10*sin(2*pi*(3*10^6)*x_0)
##########################################################################
FWHM_real = [
    771.62, 385.8, 192.9, 96.45, 48.225, 24.1125, 12.057, 
    6.02, 3.01, 1.505, 0.7525, 0.3763, 0.18815, 0.094075
]

FWHM_noise = [
    771.2, 385.4, 192.6, 96.9, 48.5, 24.2, 12.03,
    6.05, 3, 1.507, 0.759, 0.3797, 0.19, 0.1014
]

FWHM_error = [
    0.054, 0.104, 0.156, 0.467, 0.570, 0.363, 0.224, 
    0.498, 0.332, 0.132, 0.864, 0.904, 0.983, 7.786, 
]

abscissa = [
    '$2^{14}$', '$2^{15}$', '$2^{16}$', '$2^{17}$', '$2^{18}$', '$2^{19}$','$2^{20}$', 
    '$2^{21}$', '$2^{22}$', '$2^{23}$', '$2^{24}$', '$2^{25}$', '$2^{26}$', '$2^{27}$'
]

figure(1)
real, = plt.plot(abscissa, FWHM_real, marker='o', ms=5)
noise, = plt.plot(abscissa, FWHM_noise, marker='^', ms=5)
plt.legend(handles=[real, noise],labels=['FWHM(Real)', 'FWHM(Imposing Noise)'], loc='upper right')
plt.xlabel("Number of Sampling Points")
plt.ylabel("Waveform Resolution ($m^{-1}$)")
plt.suptitle("Comparison of True Resolution and Resolution after Superimposed Noise", fontsize = 25)

figure(2)
error, = plt.plot(abscissa, FWHM_error, marker='o', ms=5)
for i, txt in enumerate(FWHM_error):
    plt.annotate(txt, (abscissa[i], FWHM_error[i]), xycoords='data', xytext=(0, 8),
             textcoords='offset points')
plt.legend(handles=[error],labels=[r'$\epsilon = \frac{|FWHM\;Noise - FWHM\;Real|}{FWHM\;Real}$'], loc='upper left', fontsize=15)
plt.xlabel("Number of Sampling Points")
plt.ylabel("Absolute Error (%)")
plt.suptitle("Absolute Error of True Resolution and Resolution after Superimposed Noise", fontsize = 25)

plt.show()

