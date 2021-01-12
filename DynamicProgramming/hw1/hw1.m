clear;
x=0:0.001:10;
y1=2*sin(pi*x/3);
y2=sin(pi*x*2);
figure;
hold on;axis([0 10 -3 3]);
title('两种正弦序列')
plot(x,y1,'r');
plot(x,y2,'b');

figure;
hold on;
title('两种正弦信号相加：红色    相乘：蓝色')
plot(x,y1+y2,'r');
plot(x,y1.*y2,'b');

figure;
hold on;
title('正弦信号：红色   延时：蓝色')
y3=sin(pi*x*2+pi);
plot(x,y2,'r');
plot(x,y3,'b');

figure;
hold on;
title('正弦信号：红色   翻转+延时：蓝色')
y4=abs(sin(pi*x*2+pi/6));
plot(x,y2,'r');
plot(x,y4,'b');


