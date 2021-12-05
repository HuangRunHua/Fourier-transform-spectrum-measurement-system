import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pylab import *
import matplotlib.gridspec as gridspec
from scipy import signal
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#######################################################################################
# 本程序模拟有限扫描长度误差误差影响下的傅里叶变换光谱测量系统的光谱测量曲线
# 程序具体参数如下：
#   - 采样间隔选取 79.1nm
#   - 干涉图波长选取 532nm
#   - 干涉图采样点数选取 2^13（该点数下仿真图像最佳）
#   - 本实验只叠加一种噪声---线性噪声
#   - 实验目的在于模拟累积的线性噪声叠加后对信号的影响，同时在此基础上观察扫描长度增加对信号点影响
#######################################################################################

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
p1 = (-1)*(2**12)*79.1*10**(-9)
# 2**n个点 
p2 = (2**12-1)*79.1*10**(-9)
# 无补零 
x0 = np.arange(p1, p2, i)
x0_error = np.arange(p1, p2, i)
n0 = n1 = 2**(int(np.log2(len(x0)))+1)
print("n0 length = %d" %n0)

###################################################
#  此部分为叠加线性噪声
###################################################
I0 = np.cos(2*np.pi*sigma0*x0)
noise_linear = linspace(0,i/100,len(x0))

for j in range(1, len(x0)):
    x0_error[j] = x0_error[j-1] + noise_linear[j] + i


# 直线
Iw2 = np.cos(2*np.pi*sigma0*(x0_error))

# 直线
Y0, Yw2 = GetFFT(I0, Iw2, n0)

# 设置频谱图的横坐标
fs_2 = 1/i*np.arange(n0/2)/n0

best_Y0_average_range = 0.5*np.ones((Yw2.size, 1))
p_2 = np.arange(len(x0))

# 绘制直线

figure(1)
ax = plt.subplot(2,1,1)
axins_plot1, = plt.plot(x0)
axins_plot2, = plt.plot(x0_error)
plt.title("Real sampling points and after imposing linear noise")

plt.legend(handles=[axins_plot1, axins_plot2],labels=['Real sampling points', 'After imposing linear noise'], loc='upper left')

# 嵌入绘制局部放大图的坐标系
axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.5, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)

# 在子坐标系中绘制原始数据
axins.plot(p_2, x0)
axins.plot(p_2, x0_error)


# 设置放大区间
zone_left = 4000
zone_right = 4005

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5 # x轴显示范围的扩展比例
y_ratio = 0.5 # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = p_2[zone_left]-(p_2[zone_right]-p_2[zone_left])*x_ratio
xlim1 = p_2[zone_right]+(p_2[zone_right]-p_2[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((x0[zone_left:zone_right], x0_error[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)



subplot(2,2,3)
plt.plot(fs_2, Y0, marker='o', ms=5)
x2, = plt.plot(fs_2,best_Y0_average_range)
plt.legend(handles=[x2],labels=['FWHM = 1543.2 $m^{-1}$'], loc='upper right')
plt.title("$I_0$ Original spectrum curve")
plt.xlim(1.83*(10**6), 1.935*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')

subplot(2,2,4)
plt.plot(fs_2, Yw2, marker='o', ms=5)
x1, = plt.plot(fs_2,best_Y0_average_range)
plt.title("After superimposing linear noise")
plt.legend(handles=[x1],labels=['FWHM = 14460 $m^{-1}$'], loc='upper right')
plt.xlim(1.83*(10**6), 1.935*(10**6))
plt.xlabel('Wave number ($m^{-1}$)')


plt.suptitle("532nm - Comparison of the Original Signal and the Signal after Adding Linear Noise", fontsize = 20)


plt.show()
