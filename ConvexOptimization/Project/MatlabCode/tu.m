
x=fval(:,1);
y=fval(:,2);
z=fval(:,3);
%{
lin1=find(x==min(x));
a1(1)=lin1;
c1=1;
for i=1:30
    
    
end
%}
figure(1)

subplot(2,2,1)
plot(x,y,'ro','Markersize',10);
xlabel('�����ռ�');
ylabel('������ָ��');
set(gca,'fontsize',18);
grid on;
axis square

subplot(2,2,2)
plot(y,z,'ro','Markersize',10);
xlabel('������ָ��');
ylabel('���ؽ�����');
set(gca,'fontsize',18);
grid on;
axis square

subplot(2,2,3)
plot(x,z,'ro','Markersize',10);
xlabel('�����ռ�');
ylabel('���ؽ�����');
set(gca,'fontsize',18);
grid on;
axis square

subplot(2,2,4)
plot3(x,y,z,'ro','Markersize',10);
xlabel('�����ռ�');
ylabel('������ָ��');
zlabel('���ؽ�����');
set(gca,'fontsize',18);
grid on;
axis square

%{
figure(2)
scatter(y,z,100,x,'filled');
xlabel('ȫ������������ָ��');
ylabel('���ؽ�����');
grid on;
set(gca,'fontsize',20);
h = colorbar;
colormap(hsv);
set(get(h,'label'),'string','�����ռ�ָ��','fontsize',25);%����ɫ������
%}
