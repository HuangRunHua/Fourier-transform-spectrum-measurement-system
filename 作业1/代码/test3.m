clc;
clear;
close all;

% 波长为632.8nm
lambda=632.8*10^(-9);
% 光程差
x=0:lambda/10:2;
% 光速
c=3*10^8;
% 632.8nm波长对应的频率
f=c/lambda;
% 频间隔为300MHz
delta_f=300*10^6;
% 幅度
I=1;

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

% 将3个不同频率、5个不同频率和7个不同频率的干涉图相加
IA=I0+I1+I2;
IB=I0+I1+I2+I3+I4;
IC=I0+I1+I2+I3+I4+I5+I6;

% 画图
figure;
subplot(3,1,1);
plot(x,IA,'r.-','LineWidth',0.5);
title('632.8nm-3个频率干涉');
subplot(3,1,2);
plot(x,IB,'b.-','LineWidth',0.5);
title('632.8nm-5个频率干涉');
subplot(3,1,3);
plot(x,IC,'r.-','LineWidth',0.5);
title('632.8nm-7个频率干涉');
