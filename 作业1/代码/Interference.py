import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from pylab import *

class SuperpositionOfWaves(object):
    
    def __init__(self, lambda_center, sample_f, scan_length, sameAmplitude):
        # 设置是否等幅度
        self.sameAmplitude = sameAmplitude
        # 设置中心波长532nm
        self.lambda_center = lambda_center
        # 设置光速
        self.c = 3*10**8
        # 频率/HZ
        self.f = self.c/self.lambda_center
        # 频率间隔300MHz
        self.delta_f = 300*10**6
        # 干涉图采样间隔/m
        self.sample_f = sample_f
        # 扫描长度/m 
        self.scan_length = scan_length
        # 采样点个数
        # self.N = int(np.power(2,np.ceil(np.log2(self.scan_length/self.sample_f))))
        self.N = int(self.scan_length/self.sample_f)*2
        self.N1 = int(self.scan_length/self.sample_f)
        # 光程差/m
        # self.x = np.linspace(0,self.scan_length,int(self.scan_length/self.sample_f))
        self.x = np.linspace(0,self.scan_length,self.N)
        # 设置采样频率
        self.fs = 1/self.sample_f*np.arange(self.N)/self.N1
        # 设定强度频率/Hz
        self.f0 = self.f
        self.f1 = self.f-self.delta_f
        self.f2 = self.f+self.delta_f
        self.f3 = self.f-2*self.delta_f
        self.f4 = self.f+2*self.delta_f
        self.f5 = self.f-3*self.delta_f
        self.f6 = self.f+3*self.delta_f

        # 计算波长/m
        self.lambda0 = self.c/self.f0
        self.lambda1 = self.c/self.f1
        self.lambda2 = self.c/self.f2
        self.lambda3 = self.c/self.f3
        self.lambda4 = self.c/self.f4
        self.lambda5 = self.c/self.f5
        self.lambda6 = self.c/self.f6

        # 计算波数/m**(-1)
        self.sigma0=1/self.lambda0
        self.sigma1=1/self.lambda1
        self.sigma2=1/self.lambda2
        self.sigma3=1/self.lambda3
        self.sigma4=1/self.lambda4
        self.sigma5=1/self.lambda5
        self.sigma6=1/self.lambda6

        # 设定叠加的波
        if self.sameAmplitude:
            self.i0=1*(1+np.cos(2*np.pi*self.sigma0*self.x))
            self.i1=1*(1+np.cos(2*np.pi*self.sigma1*self.x))
            self.i2=1*(1+np.cos(2*np.pi*self.sigma2*self.x))
            self.i3=1*(1+np.cos(2*np.pi*self.sigma3*self.x))
            self.i4=1*(1+np.cos(2*np.pi*self.sigma4*self.x))
            self.i5=1*(1+np.cos(2*np.pi*self.sigma5*self.x))
            self.i6=1*(1+np.cos(2*np.pi*self.sigma6*self.x))
        else:
            self.i0=20*(1+np.cos(2*np.pi*self.sigma0*self.x))
            self.i1=15*(1+np.cos(2*np.pi*self.sigma1*self.x))
            self.i2=15*(1+np.cos(2*np.pi*self.sigma2*self.x))
            self.i3=10*(1+np.cos(2*np.pi*self.sigma3*self.x))
            self.i4=10*(1+np.cos(2*np.pi*self.sigma4*self.x))
            self.i5=5*(1+np.cos(2*np.pi*self.sigma5*self.x))
            self.i6=5*(1+np.cos(2*np.pi*self.sigma6*self.x))

        # 组合多种波
        self.ia = self.i0+self.i1+self.i2
        self.ib = self.i0+self.i1+self.i2+self.i3+self.i4
        self.ic = self.i0+self.i1+self.i2+self.i3+self.i4+self.i5+self.i6

    def drawInterferencePic(self):
        print(self.N)
        wave = [self.ia, self.ib, self.ic]
        for j in range(3):
            plt.subplot(3,3,3*j+1)
            plt.plot(self.x, wave[j])
            if j == 0:
                plt.title("Interferogram(3 waveforms)")
            if j == 1:
                plt.title("Interferogram(5 waveforms)")
            if j == 2:
                plt.title("Interferogram(7 waveforms)")

            # 快速傅立叶变换
            # y = np.abs(fft(wave[j]))/self.N
            y = 2*np.abs(fft(wave[j], self.N))/self.N
            plt.subplot(3,3,3*j+2)
            plt.plot(self.fs, y)
            plt.title("Original spectrum")

            # 放大后的频谱
            plt.subplot(3,3,3*j+3)
            plt.plot(self.fs, y)
            plt.title("Spectrogram zoom")
            if self.lambda_center < 632.8*10**(-9):
                plt.xlim(1.87967*(10**6), 1.87973*(10**6))
            else:
                plt.xlim(1.580255*(10**6), 1.5803*(10**6))
            if self.sameAmplitude:
                plt.ylim(0, 1.5)
            else:
                # plt.ylim(0,10)
                plt.ylim(0,20)
            
        if self.lambda_center < 632.8*10**(-9):
            plt.suptitle("532nm - Superposition of waves of different frequencies", fontsize = 20)  
        else:
            plt.suptitle("632.8nm - Superposition of waves of different frequencies", fontsize = 20)
        
        plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.2, 
                            hspace=0.35)
        plt.show()

a = SuperpositionOfWaves(sameAmplitude=True,lambda_center=632.8*10**(-9), sample_f=(632.8*10**(-9))/4, scan_length=2.0)
a.drawInterferencePic()