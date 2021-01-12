clc;
clear;
close all;
Fs = 1000;%采样频率
T = 1/Fs;%采样时间
L = 10240;%信号长度
t = (0:L-1)*T; %时间序列

%% 参数定义
f_n = 50;
w_n = 2*pi*f_n;%固有频率
zeta = 0.1;%阻尼比

%% 输入信号和噪声信号生成
w_d = w_n * sqrt(1-zeta^2);
h = (1/w_d)*exp(-zeta*w_n*t).*sin(w_d*t);%单位冲激响应函数h(t)

rng(10,'twister'); %设置随机生成器状态
x = randn(1,40*L); %随机输入信号x

rng(20,'twister');
n = 0.001*randn(1,40*L); %输出混杂的噪声n，方差为0.001

%% 画输入信号
figure;
subplot(2,3,1);
plot((0:40*L-1)*T,x); 
xlabel('Time','Interpreter','latex');
ylabel('x(t)','Interpreter','latex');
xlim([0,1]);
title('输入信号x');
%% 画噪声信号
subplot(2,3,2);
plot((0:40*L-1)*T,n); 
xlabel('Time','Interpreter','latex');
ylabel('n(t)','Interpreter','latex');
xlim([0,1]);
title('输出混杂的噪声信号n，默认方差为0.001');

%% 输出信号生成
y = conv(x,h); %卷积求系统输出
y = y(1:40*L);
y_m = y+n; %输出信号

%% 使用cpsd计算互功率谱和自功率谱
[Pxx,w]=cpsd(x,x,hanning(L),L/2,L,Fs);
[Pyy,w]=cpsd(y_m,y_m,hanning(L),L/2,L,Fs);
[Pxy,f]=cpsd(x,y_m,hanning(L),L/2,L,Fs);

%% 设置题目参数条件
H = fft(h); %系统理想的频率响应函数
H_0 = fft(y_m(1:L))./fft(x(1:L)); %题目中H0
H_1 = Pxy./Pxx; %题目中H1
H_2 = Pyy./conj(Pxy); %题目中H2

%% 绘制H与H0对比图
subplot(2,3,3);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_0(1:L/2+1))))
hold on;

set(legend('H(f)','H_0(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('H0与理想频响函数比较');

%% 绘制H与H1对比图
subplot(2,3,4);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_1)))
hold on;

set(legend('H(f)','H_1(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('H1与理想频响函数比较');

%% 绘制H与H2对比图
subplot(2,3,5);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_2)))
hold on;

set(legend('H(f)','H_2(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('H2与理想频响函数比较');

%% 更改噪声方差为0.01绘制H与H2对比图
rng(20,'twister');
n = 0.01*randn(1,40*L); %方差为0.01的噪声n
y_m = y+n; %新的输出信号
[Pyy,w]=cpsd(y_m,y_m,hanning(L),L/2,L,Fs);
[Pxy,f]=cpsd(x,y_m,hanning(L),L/2,L,Fs);

H = fft(h); %系统理想的频率响应函数
H_2 = Pyy./conj(Pxy); %题目中H2

subplot(2,3,6);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_2)))
hold on;

set(legend('H(f)','H_2(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('噪声方差为0.01时的H2与理想频响函数比较');
