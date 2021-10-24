% 波长
lambda0=532*10^(-9);
lambda1=632.8*10^(-9);
% 光程差
x=0:lambda0/100:5*lambda0;
% 波数
sigma0=1/lambda0;
sigma1=1/lambda1;
% 干涉谱
I0=1*(1+cos(2*pi*sigma0*x));
I1=1*(1+cos(2*pi*sigma1*x));
I3=I1+I0;
% 画图
subplot(3,1,1)
plot(x,I0,'r.-','LineWidth',0.5)
axis([0 2.5*10^(-6) 0 2])
title('532nm')
subplot(3,1,2)
plot(x,I1,'b.-','LineWidth',0.5)
axis([0 2.5*10^(-6) 0 2])
title('632.8nm')
subplot(3,1,3)
plot(x,I3,'b.-','LineWidth',0.5)
axis([0 2.5*10^(-6) 0 4])
title('532nm+632.8nm')
