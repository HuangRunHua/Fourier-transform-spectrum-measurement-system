clc;
clear;
close all;

% ����
lambda=632.8*10^-9;
% ����
c=3*10^8;
% Ƶ��
f=c/lambda;
% Ƶ���
delta_f=1000*10^6;
% ����ͼ�������
sampling_interval_f=(632.8*10^(-9))/8;
% ɨ�賤��
scan_length=2.0;

% ��̲�
x=0-scan_length/2:sampling_interval_f:scan_length/2;

% ���ò�ͬƵ��
f0=f;
f1=f-delta_f;
f2=f+delta_f;
f3=f-2*delta_f;
f4=f+2*delta_f;
f5=f-3*delta_f;
f6=f+3*delta_f;
% ��ͬƵ�ʶ�Ӧ�Ĳ���
lambda0=c/f0;
lambda1=c/f1;
lambda2=c/f2;
lambda3=c/f3;
lambda4=c/f4;
lambda5=c/f5;
lambda6=c/f6;
% ��ͬƵ�ʶ�Ӧ�Ĳ���
sigma0=1/lambda0;
sigma1=1/lambda1;
sigma2=1/lambda2;
sigma3=1/lambda3;
sigma4=1/lambda4;
sigma5=1/lambda5;
sigma6=1/lambda6;
% ��ͬƵ�ʶ�Ӧ�ĸ���ǿ��
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

% ��3����ͬƵ�ʡ�5����ͬƵ�ʺ�7����ͬƵ�ʵĸ���ͼ���
IA=I0+I1+I2;
IB=I0+I1+I2+I3+I4;
IC=I0+I1+I2+I3+I4+I5+I6;
 
% 632.8nm-3����ͬƵ�ʸ���
subplot(3,3,1); 
plot(x,IA,'r.-','LineWidth',0.5);
grid on;
xlabel('��̲�/x');
ylabel('����ǿ��I(x)');
title('632.8nm-3����ͬƵ�ʸ���');

% ���ٸ���Ҷ�任                  
N=floor(scan_length/sampling_interval_f);
%�������
ts=sampling_interval_f;
%����Ƶ��
fs=1/ts;
% ʱ������
time=fs/2*linspace(0,1,N/2+1);
Y = fft(IA,N)/N;
Y=2*abs(Y(1:floor(N/2)+1));
subplot(3,3,2);
plot(time,Y);
grid on;
xlabel('����');
ylabel('��ֵ');
title('3����ͬƵ�ʹ���ͼ'); 

subplot(3,3,3);
plot(time,Y);
grid on;
xlabel('����');
ylabel('��ֵ');
title('����ͼ�Ŵ�');
xlim([1.5802*10^6 1.58035*10^6]);

 
% 632.8nm-5����ͬƵ�ʸ���
subplot(3,3,4); 
plot(x,IB,'r.-','LineWidth',0.5);
grid on;
xlabel('��̲�/x');
ylabel('����ǿ��I(x)');
title('632.8nm-5����ͬƵ�ʸ���');

% ���ٸ���Ҷ�任                  
N=floor(scan_length/sampling_interval_f);
%�������
ts=sampling_interval_f;
%����Ƶ��
fs=1/ts;
% ʱ������
time=fs/2*linspace(0,1,N/2+1);
Y = fft(IB,N)/N;
Y=2*abs(Y(1:floor(N/2)+1));
subplot(3,3,5);
plot(time,Y);
grid on;
xlabel('����');
ylabel('��ֵ');
title('5����ͬƵ�ʹ���ͼ'); 

subplot(3,3,6);
plot(time,Y);
grid on;
xlabel('����');
ylabel('��ֵ');
title('����ͼ�Ŵ�');
xlim([1.5802*10^6 1.58035*10^6]);

 
% 632.8nm-7����ͬƵ�ʸ���
subplot(3,3,7); 
plot(x,IC,'r.-','LineWidth',0.5);
grid on;
xlabel('��̲�/x');
ylabel('����ǿ��I(x)');
title('632.8nm-7����ͬƵ�ʸ���');

% ���ٸ���Ҷ�任                  
N=floor(scan_length/sampling_interval_f);
%�������
ts=sampling_interval_f;
%����Ƶ��
fs=1/ts;
% ʱ������
time=fs/2*linspace(0,1,N/2+1);
Y = fft(IA,N)/N;
Y=2*abs(Y(1:floor(N/2)+1));
subplot(3,3,8);
plot(time,Y);
grid on;
xlabel('����');
ylabel('��ֵ');
title('7����ͬƵ�ʹ���ͼ'); 

subplot(3,3,9);
plot(time,Y);
grid on;
xlabel('����');
ylabel('��ֵ');
title('����ͼ�Ŵ�');
xlim([1.5802*10^6 1.58035*10^6]);