clc;
clear;
close all;

% 波长
lambda=632.8*10^-9;
% 光速
c=3*10^8;
% 频率
f=c/lambda;
% 频间隔
delta_f=1000*10^6;
% 干涉图采样间隔
sampling_interval_f=(632.8*10^(-9))/8;
% 扫描长度
scan_length=2.0;

% 光程差
x=0-scan_length/2:sampling_interval_f:scan_length/2;

% 设置不同频率
f0=f;
f1=f-delta_f;
f2=f+delta_f;
f3=f-2*delta_f;
f4=f+2*delta_f;
f5=f-3*delta_f;
f6=f+3*delta_f;
% 不同频率对应的波长
lambda0=c/f0;
lambda1=c/f1;
lambda2=c/f2;
lambda3=c/f3;
lambda4=c/f4;
lambda5=c/f5;
lambda6=c/f6;
% 不同频率对应的波数
sigma0=1/lambda0;
sigma1=1/lambda1;
sigma2=1/lambda2;
sigma3=1/lambda3;
sigma4=1/lambda4;
sigma5=1/lambda5;
sigma6=1/lambda6;
% 不同频率对应的干涉强度
I0=20*(1+cos(2*pi*sigma0*x));
I1=15*(1+cos(2*pi*sigma1*x));
I2=15*(1+cos(2*pi*sigma2*x));
I3=10*(1+cos(2*pi*sigma3*x));
I4=10*(1+cos(2*pi*sigma4*x));
I5=5*(1+cos(2*pi*sigma5*x));
I6=5*(1+cos(2*pi*sigma6*x));
I0=1*(1+cos(2*pi*sigma0*x));
I1=1*(1+cos(2*pi*sigma1*x));
I2=1*(1+cos(2*pi*sigma2*x));
I3=1*(1+cos(2*pi*sigma3*x));
I4=1*(1+cos(2*pi*sigma4*x));
I5=1*(1+cos(2*pi*sigma5*x));
I6=1*(1+cos(2*pi*sigma6*x));

% 将3个不同频率、5个不同频率和7个不同频率的干涉图相加
IA=I0+I1+I2;
IB=I0+I1+I2+I3+I4;
IC=I0+I1+I2+I3+I4+I5+I6;
 
% 632.8nm-3个不同频率干涉
subplot(3,3,1); 
plot(x,IA,'r.-','LineWidth',0.5);
grid on;
xlabel('光程差/x');
ylabel('干涉强度I(x)');
title('632.8nm-3个不同频率干涉');

% 快速傅里叶变换                  
N=floor(scan_length/sampling_interval_f);
%采样间隔
ts=sampling_interval_f;
%采样频率
fs=1/ts;
% 时间向量
time=fs/2*linspace(0,1,N/2+1);
Y = fft(IA,N)/N;
Y=2*abs(Y(1:floor(N/2)+1));
subplot(3,3,2);
plot(time,Y);
grid on;
xlabel('波数');
ylabel('幅值');
title('3个不同频率光谱图'); 

subplot(3,3,3);
plot(time,Y);
grid on;
xlabel('波数');
ylabel('幅值');
title('光谱图放大');
xlim([1.5802*10^6 1.58035*10^6]);

 
% 632.8nm-5个不同频率干涉
subplot(3,3,4); 
plot(x,IB,'r.-','LineWidth',0.5);
grid on;
xlabel('光程差/x');
ylabel('干涉强度I(x)');
title('632.8nm-5个不同频率干涉');

% 快速傅里叶变换                  
N=floor(scan_length/sampling_interval_f);
%采样间隔
ts=sampling_interval_f;
%采样频率
fs=1/ts;
% 时间向量
time=fs/2*linspace(0,1,N/2+1);
Y = fft(IB,N)/N;
Y=2*abs(Y(1:floor(N/2)+1));
subplot(3,3,5);
plot(time,Y);
grid on;
xlabel('波数');
ylabel('幅值');
title('5个不同频率光谱图'); 

subplot(3,3,6);
plot(time,Y);
grid on;
xlabel('波数');
ylabel('幅值');
title('光谱图放大');
xlim([1.5802*10^6 1.58035*10^6]);

 
% 632.8nm-7个不同频率干涉
subplot(3,3,7); 
plot(x,IC,'r.-','LineWidth',0.5);
grid on;
xlabel('光程差/x');
ylabel('干涉强度I(x)');
title('632.8nm-7个不同频率干涉');

% 快速傅里叶变换                  
N=floor(scan_length/sampling_interval_f);
%采样间隔
ts=sampling_interval_f;
%采样频率
fs=1/ts;
% 时间向量
time=fs/2*linspace(0,1,N/2+1);
Y = fft(IA,N)/N;
Y=2*abs(Y(1:floor(N/2)+1));
subplot(3,3,8);
plot(time,Y);
grid on;
xlabel('波数');
ylabel('幅值');
title('7个不同频率光谱图'); 

subplot(3,3,9);
plot(time,Y);
grid on;
xlabel('波数');
ylabel('幅值');
title('光谱图放大');
xlim([1.5802*10^6 1.58035*10^6]);