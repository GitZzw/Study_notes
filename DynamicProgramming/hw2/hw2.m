N=32;
b=[1 0 -3 0 -3 0 -1];
a=[6 0 0 0 -2 0 0];
%��λ�弤��Ӧ
figure;
x=[1 zeros(1,N-1)];
k=0:1:N-1;
y=filter(b,a,x);
stem(k,y);
title('��λ�弤��Ӧ')
xlabel('n');
ylabel('����')

%��λ��Ծ��Ӧ
figure;
x2=ones(1,N);
k=0:1:N-1;
y=filter(b,a,x2);
stem(k,y);
title('��λ��Ծ��Ӧ')
xlabel('n');
ylabel('����')
