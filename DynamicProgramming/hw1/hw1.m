clear;
x=0:0.001:10;
y1=2*sin(pi*x/3);
y2=sin(pi*x*2);
figure;
hold on;axis([0 10 -3 3]);
title('������������')
plot(x,y1,'r');
plot(x,y2,'b');

figure;
hold on;
title('���������ź���ӣ���ɫ    ��ˣ���ɫ')
plot(x,y1+y2,'r');
plot(x,y1.*y2,'b');

figure;
hold on;
title('�����źţ���ɫ   ��ʱ����ɫ')
y3=sin(pi*x*2+pi);
plot(x,y2,'r');
plot(x,y3,'b');

figure;
hold on;
title('�����źţ���ɫ   ��ת+��ʱ����ɫ')
y4=abs(sin(pi*x*2+pi/6));
plot(x,y2,'r');
plot(x,y4,'b');


