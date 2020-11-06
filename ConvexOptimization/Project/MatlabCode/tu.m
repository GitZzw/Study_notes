
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
xlabel('工作空间');
ylabel('灵巧性指标');
set(gca,'fontsize',18);
grid on;
axis square

subplot(2,2,2)
plot(y,z,'ro','Markersize',10);
xlabel('灵巧性指标');
ylabel('最大关节力矩');
set(gca,'fontsize',18);
grid on;
axis square

subplot(2,2,3)
plot(x,z,'ro','Markersize',10);
xlabel('工作空间');
ylabel('最大关节力矩');
set(gca,'fontsize',18);
grid on;
axis square

subplot(2,2,4)
plot3(x,y,z,'ro','Markersize',10);
xlabel('工作空间');
ylabel('灵巧性指标');
zlabel('最大关节力矩');
set(gca,'fontsize',18);
grid on;
axis square

%{
figure(2)
scatter(y,z,100,x,'filled');
xlabel('全域灵巧性评价指标');
ylabel('最大关节力矩');
grid on;
set(gca,'fontsize',20);
h = colorbar;
colormap(hsv);
set(get(h,'label'),'string','工作空间指标','fontsize',25);%给颜色栏命名
%}
