import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *
from scipy import signal


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

# 设置中心波长为632.8nm
laimda0 = 632.8*10**(-9) 
#用79.1nm的采样间隔
i = 79.1*10**(-9) 
# 设置中心点的采样频率
sigma0 = 1/laimda0  

p1 = (-1)*(2**10)*79.1*10**(-9)
p2 = (2**10)*79.1*10**(-9)
p3 = (2**11)
p4 = 2**10

#################################
# 设置干涉图的采样点
#（无补零的情况）
#################################
x0 = np.arange(p1, p2, i)
# 计算采样点的个数 
print("没有补零前x0的长度为 %d" %len(x0))      
# 做傅里叶的点数，取下一个2的次幂
# n0 = 2**(int(np.log2(len(x0)))+1)
n0 = len(x0)
print("n0 length = %d" %n0)
# 干涉图函数
I0=(np.cos(2*np.pi*sigma0*x0))
print("没有补零前I0的长度为 %d" %len(I0)) 

# 这里对I0做n0个点的FFT，为了求得正确的幅度需要除以len(x0)
# 真实的点个数，不是除以傅立叶变换的点数
Y0 = 2*abs(fft(I0,n0))/len(x0)
# 得到的Y0是周期函数，为了使用 FWHM函数需要提取出前一半图像
Y0 = Y0[:int(n0/2)]

# 设置频谱图的横坐标
fs = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Y0.size, 1))

print("------------------------------------")

#################################################################
# Zero padding begins here
# Case 1.
#   1*2^11 zeros
#################################################################
zero_numbers_1= p3/2
x1 = np.arange((-1)*(p4+zero_numbers_1)*i, (p4+zero_numbers_1)*i, i)
print("补零后x1的长度为 %d" %len(x1)) 
# 做傅里叶的点数，取下一个2的次幂
n1 = 2**(int(np.log2(len(x1)))+1)
print("n1 length = %d" %n1)
I1=(np.cos(2*np.pi*sigma0*x0));

I1 = np.pad(I1,(0,int(zero_numbers_1)),'constant',constant_values = (0,0))
I1 = np.pad(I1,(int(zero_numbers_1),0),'constant',constant_values = (0,0))
print("补零后I1的长度为 %d" %len(I1)) 


Y1 = 2*abs(fft(I1,n1))/len(x0)
Y1 = Y1[:int(n1/2)]


# 设置频谱图的横坐标
fs_1 = 1/i*np.arange(n1/2)/n1

best_Y1_average_range = 0.5*np.ones((Y1.size, 1))

figure(1)
subplot(2,2,1)
plt.plot(x0, I0)
plt.title("Original waveform(No zero padding)")

subplot(2,2,2)
plt.plot(fs, Y0)
plt.plot(fs,best_Y0_average_range)
plt.title("Spectrogram(No zero padding)  FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

subplot(2,2,3)
plt.plot(x1, I1)
plt.title("Original waveform($2^{11}$ zeros padding)")

subplot(2,2,4)
plt.plot(fs_1, Y1)
plt.plot(fs_1,best_Y1_average_range)
plt.title("Spectrogram($2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves", fontsize = 20)

print("------------------------------------")

#################################################################
# Zero padding case 2.
#   3*2^11 zeros
#################################################################
zero_numbers_2= 3*p3/2
x2 = np.arange((-1)*(p4+zero_numbers_2)*i, (p4+zero_numbers_2)*i, i)
print("补零后x1的长度为 %d" %len(x1)) 
# 做傅里叶的点数，取下一个2的次幂
n2 = 2**(int(np.log2(len(x2)))+1)
print("n2 length = %d" %n2)
I2=(np.cos(2*np.pi*sigma0*x0));

I2 = np.pad(I2,(0,int(zero_numbers_2)),'constant',constant_values = (0,0))
I2 = np.pad(I2,(int(zero_numbers_2),0),'constant',constant_values = (0,0))
print("补零后I2的长度为 %d" %len(I2)) 


Y2 = 2*abs(fft(I2,n2))/len(x0)
Y2 = Y2[:int(n2/2)]

# 设置频谱图的横坐标
fs_2 = 1/i*np.arange(n2/2)/n2

best_Y2_average_range = 0.5*np.ones((Y2.size, 1))

figure(2)
subplot(2,2,1)
plt.plot(x0, I0)
plt.title("Original waveform(No zero padding)")

subplot(2,2,2)
plt.plot(fs, Y0)
plt.plot(fs,best_Y0_average_range)
plt.title("Spectrogram(No zero padding)  FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

subplot(2,2,3)
plt.plot(x2, I2)
plt.title("Original waveform($3*2^{11}$ zeros padding)")

subplot(2,2,4)
plt.plot(fs_2, Y2)
plt.plot(fs_2,best_Y2_average_range)
plt.title("Spectrogram($3*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves", fontsize = 20)

print("------------------------------------")

#################################################################
# Zero padding case 3.
#   5*2^11 zeros
#################################################################
zero_numbers_3= 7*p3/2
x3 = np.arange((-1)*(p4+zero_numbers_3)*i, (p4+zero_numbers_3)*i, i)
print("补零后x1的长度为 %d" %len(x3)) 
# 做傅里叶的点数，取下一个2的次幂
n3 = 2**(int(np.log2(len(x3)))+1)
print("n3 length = %d" %n1)
I3=(np.cos(2*np.pi*sigma0*x0));

I3 = np.pad(I3,(0,int(zero_numbers_3)),'constant',constant_values = (0,0))
I3 = np.pad(I3,(int(zero_numbers_3),0),'constant',constant_values = (0,0))
print("补零后I3的长度为 %d" %len(I3)) 


Y3 = 2*abs(fft(I3,n3))/len(x0)
Y3 = Y3[:int(n3/2)]

# 设置频谱图的横坐标
fs_3 = 1/i*np.arange(n3/2)/n3

best_Y3_average_range = 0.5*np.ones((Y3.size, 1))

figure(3)
subplot(2,2,1)
plt.plot(x0, I0)
plt.title("Original waveform(No zero padding)")

subplot(2,2,2)
plt.plot(fs, Y0)
plt.plot(fs,best_Y0_average_range)
plt.title("Spectrogram(No zero padding)  FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

subplot(2,2,3)
plt.plot(x3, I3)
plt.title("Original waveform($7*2^{11}$ zeros padding)")

subplot(2,2,4)
plt.plot(fs_3, Y3)
plt.plot(fs_3,best_Y3_average_range)
plt.title("Spectrogram($7*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves", fontsize = 20)

figure(10)
pic1, = plt.plot(fs, Y0)
pic2, = plt.plot(fs_1, Y1)
pic3, = plt.plot(fs_2, Y2)
pic4, = plt.plot(fs_3, Y3)
plt.title("Spectrogram All in One Picture")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.legend(handles=[pic1,pic2,pic3,pic4],labels=['No zero padding', '$1*2^{11}$ zeros padding', '$3*2^{11}$ zeros padding', '$7*2^{11}$ zeros padding'], loc='upper right')


plt.suptitle("632.8nm - Superposition of waves", fontsize = 20)


#######################################################
# Window function begins here
# Case1.
#   - hamming
#######################################################

zero_numbers_3= 7*p3/2
x3_hamming = np.arange((-1)*(p4+zero_numbers_3)*i, (p4+zero_numbers_3)*i, i)
print("补零后x3的长度为 %d" %len(x3)) 
# 做傅里叶的点数，取下一个2的次幂
n3_hamming = 2**(int(np.log2(len(x3_hamming)))+1)
print("n3 length = %d" %n1)
I3_hamming=(np.cos(2*np.pi*sigma0*x0))
L = len(I3_hamming)

# 定义hamming窗函数
hamming_window = np.hamming(len(I3_hamming))
# 对 x3加窗
I3_hamming = I3_hamming * hamming_window

I3_hamming = np.pad(I3_hamming,(0,int(zero_numbers_3)),'constant',constant_values = (0,0))
I3_hamming = np.pad(I3_hamming,(int(zero_numbers_3),0),'constant',constant_values = (0,0))
print("补零后I3的长度为 %d" %len(I3)) 



Y3_hamming = 2*abs(fft(I3_hamming,n3_hamming))
Y3_hamming = Y3_hamming/max(Y3_hamming)
Y3_hamming = Y3_hamming[:int(n3/2)]

# 设置频谱图的横坐标
fs_3_hamming = 1/i*np.arange(n3_hamming/2)/n3_hamming

Y3_hamming_average = (Y3_hamming.min()+Y3_hamming.max())/2
print("Y3_hamming min = %24e" %Y3_hamming.min())
print("Y3_hamming max = %24e" %Y3_hamming.max())
print("Y3_hamming_average = %24e" %Y3_hamming_average)

best_Y3_hamming_average_value = find_best_I_average_value(Y3_hamming, Y3_hamming_average)
print("best_Y3_hamming_average_value = %24e" %best_Y3_hamming_average_value)

best_Y3_hamming_average_range = best_Y3_hamming_average_value*np.ones((Y3_hamming.size, 1))
# best_Y3_hamming_average_range = 0.5*np.ones((Y3_hamming.size, 1))

first_index, last_index = find_index(best_Y3_hamming_average_range, Y3_hamming)
hamming_FWHM = fs_3_hamming[last_index] - fs_3_hamming[first_index]
print("Hamming FWHM = %d" %int(hamming_FWHM))

figure(4)
subplot(2,2,1)
plt.plot(x3, I3)
plt.title("Original waveform($7*2^{11}$ zeros padding)")

subplot(2,2,2)
plt.plot(fs_3, Y3)
plt.plot(fs_3,best_Y3_average_range)
plt.title("Spectrogram($7*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves(Hamming window)", fontsize = 20)


subplot(2,2,3)
plt.plot(x3_hamming, I3_hamming)
plt.title("Original waveform(Hamming window)")

subplot(2,2,4)
plt.plot(fs_3_hamming, Y3_hamming)
plt.plot(fs_3_hamming,best_Y3_hamming_average_range)
plt.title("Spectrogram(Hamming window) FWHM = %d $m^{-1}$" %int(hamming_FWHM))
plt.xlim(1.50*(10**6), 1.65*(10**6))


#######################################################
# Window function begins here
# Case2.
#   - kaiser
#######################################################

zero_numbers_3= 7*p3/2
x3_kaiser = np.arange((-1)*(p4+zero_numbers_3)*i, (p4+zero_numbers_3)*i, i)
print("补零后x3的长度为 %d" %len(x3)) 
# 做傅里叶的点数，取下一个2的次幂
n3_kaiser = 2**(int(np.log2(len(x3_kaiser)))+1)
print("n3 length = %d" %n3_kaiser)
I3_kaiser=(np.cos(2*np.pi*sigma0*x0));
L = len(I3_kaiser)

# 定义kaiser窗函数
kaiser_window = np.kaiser(len(I3_kaiser), 14)

# 对 x3加窗
I3_kaiser = I3_kaiser * kaiser_window

I3_kaiser = np.pad(I3_kaiser,(0,int(zero_numbers_3)),'constant',constant_values = (0,0))
I3_kaiser = np.pad(I3_kaiser,(int(zero_numbers_3),0),'constant',constant_values = (0,0))
print("补零后I3的长度为 %d" %len(I3)) 



Y3_kaiser = 2*abs(fft(I3_kaiser,n3_kaiser))
Y3_kaiser = Y3_kaiser/max(Y3_kaiser)
Y3_kaiser = Y3_kaiser[:int(n3/2)]

# 设置频谱图的横坐标
fs_3_kaiser = 1/i*np.arange(n3_kaiser/2)/n3_kaiser

Y3_kaiser_average = (Y3_kaiser.min()+Y3_kaiser.max())/2
print("Y3_kaiser min = %24e" %Y3_kaiser.min())
print("Y3_kaiser max = %24e" %Y3_kaiser.max())
print("Y3_kaiser_average = %24e" %Y3_kaiser_average)

best_Y3_kaiser_average_value = find_best_I_average_value(Y3_kaiser, Y3_kaiser_average)
print("best_Y3_kaiser_average_value = %24e" %best_Y3_kaiser_average_value)

best_Y3_kaiser_average_range = best_Y3_kaiser_average_value*np.ones((Y3_kaiser.size, 1))
# best_Y3_hamming_average_range = 0.5*np.ones((Y3_hamming.size, 1))

first_index, last_index = find_index(best_Y3_kaiser_average_range, Y3_kaiser)
kaiser_FWHM = fs_3_kaiser[last_index] - fs_3_kaiser[first_index]
print("kaiser FWHM = %d" %int(kaiser_FWHM))

figure(5)
subplot(2,2,1)
plt.plot(x3, I3)
plt.title("Original waveform($7*2^{11}$ zeros padding)")

subplot(2,2,2)
plt.plot(fs_3, Y3)
plt.plot(fs_3,best_Y3_average_range)
plt.title("Spectrogram($7*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves(Kaiser window)", fontsize = 20)


subplot(2,2,3)
plt.plot(x3_kaiser, I3_kaiser)
plt.title("Original waveform(Kaiser window)")

subplot(2,2,4)
plt.plot(fs_3_kaiser, Y3_kaiser)
plt.plot(fs_3_kaiser,best_Y3_kaiser_average_range)
plt.title("Spectrogram(Kaiser window) FWHM = %d $m^{-1}$" %int(kaiser_FWHM))
plt.xlim(1.50*(10**6), 1.65*(10**6))


#######################################################
# Window function begins here
# Case3.
#   - hanning
#######################################################

zero_numbers_3= 7*p3/2
x3_hanning = np.arange((-1)*(p4+zero_numbers_3)*i, (p4+zero_numbers_3)*i, i)
print("补零后x3的长度为 %d" %len(x3_hanning)) 
# 做傅里叶的点数，取下一个2的次幂
n3_hanning = 2**(int(np.log2(len(x3_hanning)))+1)
print("n3 length = %d" %n3_hanning)
I3_hanning=(np.cos(2*np.pi*sigma0*x0));
L = len(I3_hanning)

# 定义kaiser窗函数
hanning_window = np.hanning(len(I3_hanning))

# 对 x3加窗
I3_hanning = I3_hanning * hanning_window

I3_hanning = np.pad(I3_hanning,(0,int(zero_numbers_3)),'constant',constant_values = (0,0))
I3_hanning = np.pad(I3_hanning,(int(zero_numbers_3),0),'constant',constant_values = (0,0))
print("补零后I3的长度为 %d" %len(I3_hanning)) 



Y3_hanning = 2*abs(fft(I3_hanning,n3_hanning))
Y3_hanning = Y3_hanning/max(Y3_hanning)
Y3_hanning = Y3_hanning[:int(n3/2)]

# 设置频谱图的横坐标
fs_3_hanning = 1/i*np.arange(n3_hanning/2)/n3_hanning

Y3_hanning_average = (Y3_hanning.min()+Y3_hanning.max())/2
print("Y3_hanning min = %24e" %Y3_hanning.min())
print("Y3_hanning max = %24e" %Y3_hanning.max())
print("Y3_hanning_average = %24e" %Y3_hanning_average)

best_Y3_hanning_average_value = find_best_I_average_value(Y3_hanning, Y3_hanning_average)
print("best_Y3_hanning_average_value = %24e" %best_Y3_hanning_average_value)

best_Y3_hanning_average_range = best_Y3_hanning_average_value*np.ones((Y3_hanning.size, 1))
# best_Y3_hamming_average_range = 0.5*np.ones((Y3_hamming.size, 1))

first_index, last_index = find_index(best_Y3_hanning_average_range, Y3_hanning)
hanning_FWHM = fs_3_hanning[last_index] - fs_3_hanning[first_index]
print("hanning FWHM = %d" %int(hanning_FWHM))

figure(6)
subplot(2,2,1)
plt.plot(x3, I3)
plt.title("Original waveform($7*2^{11}$ zeros padding)")

subplot(2,2,2)
plt.plot(fs_3, Y3)
plt.plot(fs_3,best_Y3_average_range)
plt.title("Spectrogram($7*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves(Hanning window)", fontsize = 20)


subplot(2,2,3)
plt.plot(x3_hanning, I3_hanning)
plt.title("Original waveform(Hanning window)")

subplot(2,2,4)
plt.plot(fs_3_hanning, Y3_hanning)
plt.plot(fs_3_hanning,best_Y3_hanning_average_range)
plt.title("Spectrogram(Hanning window) FWHM = %d $m^{-1}$" %int(hanning_FWHM))
plt.xlim(1.50*(10**6), 1.65*(10**6))


#######################################################
# Window function begins here
# Case4.
#   - blackman
#######################################################
zero_numbers_3= 7*p3/2
x3_blackman = np.arange((-1)*(p4+zero_numbers_3)*i, (p4+zero_numbers_3)*i, i)
print("补零后x3的长度为 %d" %len(x3)) 
# 做傅里叶的点数，取下一个2的次幂
n3_blackman = 2**(int(np.log2(len(x3_blackman)))+1)
I3_blackman=(np.cos(2*np.pi*sigma0*x0));
L = len(I3_blackman)

# 定义blackman窗函数
blackman_window = np.blackman(len(I3_blackman))
# 对 x3加窗
I3_blackman = I3_blackman * blackman_window

I3_blackman = np.pad(I3_blackman,(0,int(zero_numbers_3)),'constant',constant_values = (0,0))
I3_blackman = np.pad(I3_blackman,(int(zero_numbers_3),0),'constant',constant_values = (0,0))
print("补零后I3的长度为 %d" %len(I3)) 



Y3_blackman = 2*abs(fft(I3_blackman,n3_blackman))
Y3_blackman = Y3_blackman/max(Y3_blackman)
Y3_blackman = Y3_blackman[:int(n3/2)]

# 设置频谱图的横坐标
fs_3_blackman = 1/i*np.arange(n3_blackman/2)/n3_blackman

Y3_blackman_average = (Y3_blackman.min()+Y3_blackman.max())/2
print("Y3_blackman min = %24e" %Y3_blackman.min())
print("Y3_blackman max = %24e" %Y3_blackman.max())
print("Y3_blackman_average = %24e" %Y3_blackman_average)

best_Y3_blackman_average_value = find_best_I_average_value(Y3_blackman, Y3_blackman_average)
print("best_Y3_blackman_average_value = %24e" %best_Y3_blackman_average_value)

best_Y3_blackman_average_range = best_Y3_blackman_average_value*np.ones((Y3_blackman.size, 1))
# best_Y3_blackman_average_range = 0.5*np.ones((Y3_blackman.size, 1))

first_index, last_index = find_index(best_Y3_blackman_average_range, Y3_blackman)
blackman_FWHM = fs_3_blackman[last_index] - fs_3_blackman[first_index]
print("blackman FWHM = %d" %int(blackman_FWHM))

figure(7)
subplot(2,2,1)
plt.plot(x3, I3)
plt.title("Original waveform($7*2^{11}$ zeros padding)")

subplot(2,2,2)
plt.plot(fs_3, Y3)
plt.plot(fs_3,best_Y3_average_range)
plt.title("Spectrogram($7*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves(Blackman window)", fontsize = 20)


subplot(2,2,3)
plt.plot(x3_blackman, I3_blackman)
plt.title("Original waveform(Blackman window)")

subplot(2,2,4)
plt.plot(fs_3_blackman, Y3_blackman)
plt.plot(fs_3_blackman,best_Y3_blackman_average_range)
plt.title("Spectrogram(Blackman window) FWHM = %d $m^{-1}$" %int(blackman_FWHM))
plt.xlim(1.50*(10**6), 1.65*(10**6))




#######################################################
# Window function begins here
# Case5.
#   - bartlett
#######################################################

zero_numbers_3= 7*p3/2
x3_bartlett = np.arange((-1)*(p4+zero_numbers_3)*i, (p4+zero_numbers_3)*i, i)
print("补零后x3的长度为 %d" %len(x3_bartlett)) 
# 做傅里叶的点数，取下一个2的次幂
n3_bartlett = 2**(int(np.log2(len(x3_bartlett)))+1)
print("n3 length = %d" %n3_bartlett)
I3_bartlett=(np.cos(2*np.pi*sigma0*x0));
L = len(I3_bartlett)

# 定义kaiser窗函数
bartlett_window = np.bartlett(len(I3_bartlett))

# 对 x3加窗
I3_bartlett = I3_bartlett * bartlett_window

I3_bartlett = np.pad(I3_bartlett,(0,int(zero_numbers_3)),'constant',constant_values = (0,0))
I3_bartlett = np.pad(I3_bartlett,(int(zero_numbers_3),0),'constant',constant_values = (0,0))
print("补零后I3的长度为 %d" %len(I3_bartlett)) 



Y3_bartlett = 2*abs(fft(I3_bartlett,n3_bartlett))
Y3_bartlett = Y3_bartlett/max(Y3_bartlett)
Y3_bartlett = Y3_bartlett[:int(n3/2)]

# 设置频谱图的横坐标
fs_3_bartlett = 1/i*np.arange(n3_bartlett/2)/n3_bartlett

Y3_bartlett_average = (Y3_bartlett.min()+Y3_bartlett.max())/2
print("Y3_bartlett min = %24e" %Y3_bartlett.min())
print("Y3_bartlett max = %24e" %Y3_bartlett.max())
print("Y3_bartlett_average = %24e" %Y3_bartlett_average)

best_Y3_bartlett_average_value = find_best_I_average_value(Y3_bartlett, Y3_bartlett_average)
print("best_Y3_bartlett_average_value = %24e" %best_Y3_bartlett_average_value)

best_Y3_bartlett_average_range = best_Y3_bartlett_average_value*np.ones((Y3_bartlett.size, 1))
# best_Y3_blackman_average_range = 0.5*np.ones((Y3_blackman.size, 1))

first_index, last_index = find_index(best_Y3_bartlett_average_range, Y3_bartlett)
bartlett_FWHM = fs_3_bartlett[last_index] - fs_3_bartlett[first_index]
print("bartlett FWHM = %d" %int(bartlett_FWHM))

figure(8)
subplot(2,2,1)
plt.plot(x3, I3)
plt.title("Original waveform($7*2^{11}$ zeros padding)")

subplot(2,2,2)
plt.plot(fs_3, Y3)
plt.plot(fs_3,best_Y3_average_range)
plt.title("Spectrogram($7*2^{11}$ zeros padding) FWHM = 7400 $m^{-1}$")
plt.xlim(1.50*(10**6), 1.65*(10**6))

plt.suptitle("632.8nm - Superposition of waves(Bartlett window)", fontsize = 20)


subplot(2,2,3)
plt.plot(x3_bartlett, I3_bartlett)
plt.title("Original waveform(Bartlett window)")

subplot(2,2,4)
plt.plot(fs_3_bartlett, Y3_bartlett)
plt.plot(fs_3_bartlett,best_Y3_bartlett_average_range)
plt.title("Spectrogram(Bartlett window) FWHM = %d $m^{-1}$" %int(bartlett_FWHM))
plt.xlim(1.50*(10**6), 1.65*(10**6))


########################################################
# 将所有加窗后波形汇总在一个图里面
########################################################
figure(9)
l1, = plt.plot(fs_3, Y3)
l2, = plt.plot(fs_3_hanning, Y3_hanning)
l3, = plt.plot(fs_3_kaiser, Y3_kaiser)
l4, = plt.plot(fs_3_hamming, Y3_hamming)
l5, = plt.plot(fs_3_bartlett, Y3_bartlett)
l6, = plt.plot(fs_3_blackman, Y3_blackman)
l7, = plt.plot(fs_3,best_Y3_average_range, linewidth = 2.5)
plt.xlim(1.50*(10**6), 1.65*(10**6))


plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7],labels=['Rectangular window','Hanning window', 'Kaiser window','Hamming window','Bartlett window','Blackman window', 'FWHM'], loc='upper right')

plt.suptitle("632.8nm - Superposition of waves($7*2^{11}$ zeros padding)", fontsize = 20)



plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
plt.show()