"""
本程序主要模拟扫描长度与发散角的不同带给波形的影响
本实验选取的扫描长度为0.01cm，0.1cm与2cm
本实验发散角选取为0.01pi，0.02pi与0.03pi，波长选用为532nm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *
import matplotlib.animation as animation
from scipy.interpolate import UnivariateSpline
from scipy import spatial


def fft_convolve(a,b):
    n = len(a) + len(b) - 1
    N = 2**(int(np.log2(n))+1)
    A = fft(a, N)
    B = fft(b, N)
    return ifft(A*B)[:n]

def find_best_I_average_value(array, average):
    best_I_average_value = 0
    min_dif = 1
    for i in array:
        if abs(i-average) < min_dif:
            min_dif = abs(i-average)
            best_I_average_value = i

    return best_I_average_value

def find_index(array1, array2):
    first_index = 0
    last_index = 0
    current_index = 0
    for i, j in zip(array1,array2):
        if i < j:
            first_index = current_index
            break
        current_index = current_index + 1
    array2 = array2[::-1]
    array1 = array1[::-1]
    current_index = 0
    for i, j in zip(array1,array2):
        if i < j:
            last_index = current_index
            break
        current_index = current_index + 1
    print("array1.size - last_index - first_index = %d" %(array1.size - last_index - first_index))
    return first_index, array1.size - last_index

    

N = 200000

# 设置扫描长度(m)
L1 = 0.0001
L2 = 0.001
L3 = 0.02

# 设置三个入射发散角
# sita1=0.1*np.pi 
# sita2=0.2*np.pi
# sita3=0.4*np.pi
sita1=0.01*np.pi 
sita2=0.02*np.pi
sita3=0.03*np.pi

# 设置三个入射发散角的立体角
W1=2*np.pi*(1-np.cos(sita1))
W2=2*np.pi*(1-np.cos(sita2))
W3=2*np.pi*(1-np.cos(sita3))

# 设置波长
L = 532*10**(-9)

sigma0 = 1/L
sigma1_1 = sigma0 - sigma0*W1/(100*np.pi)
sigma1_2 = sigma0 - sigma0*W2/(100*np.pi)
sigma1_3 = sigma0 - sigma0*W3/(100*np.pi)
sigma2 = sigma0

# 设置波束范围
sigma = np.arange(sigma0 - 10**5, sigma2 + 10**5)
# sigma_jiehe = np.arange(-10**4, 10**4, 0.1)
sigma_jiehe = np.arange(sigma0-10**5, sigma2+10**5)

# 入射角的影响
B1 = np.pi/(sigma0*W1)*((sigma>=sigma1_1)&(sigma<=sigma2))
B2 = np.pi/(sigma0*W2)*((sigma>=sigma1_2)&(sigma<=sigma2))
B3 = np.pi/(sigma0*W3)*((sigma>=sigma1_3)&(sigma<=sigma2))
# print(len(B3))

# 扫描长度的影响 
Y01 = 2*L1*np.sinc(2*np.pi*(sigma0-sigma)*L1) 
Y02 = 2*L2*np.sinc(2*np.pi*(sigma0-sigma)*L2) 
Y03 = 2*L3*np.sinc(2*np.pi*(sigma0-sigma)*L3)
# print(len(Y03))


sigma_jiehe1 = np.linspace(sigma0-10**5, sigma2+10**5, len(B3)+len(Y03)-1)


###############################################
# B1, Y01
# I0类型为 numpy.ndarry, 399999 x 1的向量
# 求取I0的最大值与最小值时候需要注意保留有效数字
# 最小值可以达到10^(-24)数量级，最大值才10^(-7)数量级
###############################################
I0 = fft_convolve(B1, Y01)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,1)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("0.01cm-0.01pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})


###############################################
# B1, Y02
###############################################
I0 = fft_convolve(B1, Y02)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,2)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("0.1cm-0.01pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})


###############################################
# B1, Y03
###############################################
I0 = fft_convolve(B1, Y03)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,3)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("2cm-0.01pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})

###############################################
# B2, Y01
###############################################
I0 = fft_convolve(B2, Y01)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,4)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("0.01cm-0.02pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})

###############################################
# B2, Y02
###############################################
I0 = fft_convolve(B2, Y02)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,5)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("0.1cm-0.02pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})

###############################################
# B2, Y03
###############################################
I0 = fft_convolve(B2, Y03)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,6)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("2cm-0.02pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})

###############################################
# B3, Y01
###############################################
I0 = fft_convolve(B3, Y01)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,7)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("0.01cm-0.03pi  FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})

###############################################
# B3, Y01
###############################################
I0 = fft_convolve(B3, Y02)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,8)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("0.1cm-0.03pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 20, bbox = {'facecolor':'yellow', 'alpha': 0.2})

###############################################
# B3, Y01
###############################################
I0 = fft_convolve(B3, Y03)
I0 = abs(I0)
print(I0)
I0_average = (I0.min()+I0.max())/2

best_I0_average_value = find_best_I_average_value(I0, I0_average)

best_I0_average_range = best_I0_average_value*np.ones((I0.size, 1))

first_index, last_index = find_index(best_I0_average_range, I0)
FWHM = sigma_jiehe1[last_index] - sigma_jiehe1[first_index]

subplot(3,3,9)
plt.plot(sigma_jiehe1,I0)
plt.plot(sigma_jiehe1,best_I0_average_range)
plt.xlim(1.87*10**6,1.8940*10**6)
plt.title("2cm-0.03pi      FWHM = %d m" %int(FWHM), fontsize = 10)
# plt.text(1.563*10**6,1.50*10**(-6),"FWHM = %lf m" %FWHM, fontsize = 10, bbox = {'facecolor':'yellow', 'alpha': 0.2})


plt.suptitle("532nm - Spectral measurement curve under the influence of two factors", fontsize = 20) 
plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.2, 
                            hspace=0.35)
plt.show()

