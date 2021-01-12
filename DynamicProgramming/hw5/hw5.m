clc;
clear;
close all;
Fs = 1000;%����Ƶ��
T = 1/Fs;%����ʱ��
L = 10240;%�źų���
t = (0:L-1)*T; %ʱ������

%% ��������
f_n = 50;
w_n = 2*pi*f_n;%����Ƶ��
zeta = 0.1;%�����

%% �����źź������ź�����
w_d = w_n * sqrt(1-zeta^2);
h = (1/w_d)*exp(-zeta*w_n*t).*sin(w_d*t);%��λ�弤��Ӧ����h(t)

rng(10,'twister'); %�������������״̬
x = randn(1,40*L); %��������ź�x

rng(20,'twister');
n = 0.001*randn(1,40*L); %������ӵ�����n������Ϊ0.001

%% �������ź�
figure;
subplot(2,3,1);
plot((0:40*L-1)*T,x); 
xlabel('Time','Interpreter','latex');
ylabel('x(t)','Interpreter','latex');
xlim([0,1]);
title('�����ź�x');
%% �������ź�
subplot(2,3,2);
plot((0:40*L-1)*T,n); 
xlabel('Time','Interpreter','latex');
ylabel('n(t)','Interpreter','latex');
xlim([0,1]);
title('������ӵ������ź�n��Ĭ�Ϸ���Ϊ0.001');

%% ����ź�����
y = conv(x,h); %�����ϵͳ���
y = y(1:40*L);
y_m = y+n; %����ź�

%% ʹ��cpsd���㻥�����׺��Թ�����
[Pxx,w]=cpsd(x,x,hanning(L),L/2,L,Fs);
[Pyy,w]=cpsd(y_m,y_m,hanning(L),L/2,L,Fs);
[Pxy,f]=cpsd(x,y_m,hanning(L),L/2,L,Fs);

%% ������Ŀ��������
H = fft(h); %ϵͳ�����Ƶ����Ӧ����
H_0 = fft(y_m(1:L))./fft(x(1:L)); %��Ŀ��H0
H_1 = Pxy./Pxx; %��Ŀ��H1
H_2 = Pyy./conj(Pxy); %��Ŀ��H2

%% ����H��H0�Ա�ͼ
subplot(2,3,3);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_0(1:L/2+1))))
hold on;

set(legend('H(f)','H_0(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('H0������Ƶ�캯���Ƚ�');

%% ����H��H1�Ա�ͼ
subplot(2,3,4);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_1)))
hold on;

set(legend('H(f)','H_1(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('H1������Ƶ�캯���Ƚ�');

%% ����H��H2�Ա�ͼ
subplot(2,3,5);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_2)))
hold on;

set(legend('H(f)','H_2(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('H2������Ƶ�캯���Ƚ�');

%% ������������Ϊ0.01����H��H2�Ա�ͼ
rng(20,'twister');
n = 0.01*randn(1,40*L); %����Ϊ0.01������n
y_m = y+n; %�µ�����ź�
[Pyy,w]=cpsd(y_m,y_m,hanning(L),L/2,L,Fs);
[Pxy,f]=cpsd(x,y_m,hanning(L),L/2,L,Fs);

H = fft(h); %ϵͳ�����Ƶ����Ӧ����
H_2 = Pyy./conj(Pxy); %��Ŀ��H2

subplot(2,3,6);
plot(f,20*log10(abs(H(1:L/2+1))))
hold on;

plot(f,20*log10(abs(H_2)))
hold on;

set(legend('H(f)','H_2(f)'),'Interpreter','latex');
xlabel('Frequency(HZ)','Interpreter','latex');
ylabel('|H(f)|(dB)','Interpreter','latex');
xlim([10,150]);
title('��������Ϊ0.01ʱ��H2������Ƶ�캯���Ƚ�');
