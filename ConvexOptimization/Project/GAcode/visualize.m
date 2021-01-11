figure;
plot3(fval(:,1),fval(:,2),fval(:,3),'*');
grid on;

x=fval(:,1);
y=fval(:,2);
z=fval(:,3);

figure;

subplot(1,3,1)
plot(x,y,'*','Markersize',3);
xlabel('work space(-g1)');
ylabel('dexterity(-g2)');
set(gca,'fontsize',18);
grid on;
axis square

subplot(1,3,2)
plot(y,z,'*','Markersize',3);
xlabel('dexterity(-g2)');
ylabel('torque(g3)');
set(gca,'fontsize',18);
grid on;
axis square

subplot(1,3,3)
plot(x,z,'*','Markersize',3);
xlabel('work space(-g1)');
ylabel('torque(g3)');
set(gca,'fontsize',18);
grid on;
axis square

% subplot(2,2,4)
% plot3(x,y,z,'*','Markersize',3);
% xlabel('work space');
% ylabel('dexterity');
% zlabel('torque');
% set(gca,'fontsize',18);
% grid on;
% axis square

