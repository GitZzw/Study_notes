% https://blog.csdn.net/sinat_34054843/article/details/76451504

% https://blog.csdn.net/Weizhiyuan37927/article/details/77622136
clear
clc
fitnessfcn=@linf;   %适应度函数句柄
nonlcon=@nonf3;
nvars=11;        %变量个数
lb=[0,0,0,0,0,0,0,0,0,0,0];      %下限
ub=[90,90,180,180,180,180,180,180,360,360,360];     %上限
A1=[zeros(1,8),1,-1,0;zeros(1,8),0,1,-1;zeros(1,8),-1,0,1];b=[-40;-40;320];     %线性不等式约束
Aeq=[];beq=[];  %线性等式约束
options=gaoptimset('paretoFraction',0.4,'populationsize',100,'generations',300,'stallGenLimit',300,'TolFun',1e-100,...
'CrossoverFraction',0.95,'MigrationFraction',0.9,'PlotFcns',{@gaplotpareto});
% 最优个体系数paretoFraction为0.3；种群大小populationsize为100，最大进化代数generations为200，
% 停止代数stallGenLimit为200， 适应度函数偏差TolFun设为1e-100，函数gaplotpareto：绘制Pareto前端
%[x,fval]=gamultiobj(fitnessfcn,nvars,A1,b,Aeq,beq,lb,ub,nonlcon,options);
[x,fval]=gamultiobj(fitnessfcn,nvars,A1,b,Aeq,beq,lb,ub,options);



%plot3(fval(:,1),fval(:,2),fval(:,3),'*');
%plot(fval(:,2),fval(:,3),'*')
%{
x1=find(fval(:,1) == -23)
y1=fval(x1,:);
xx=x(x1,:);
plot(y1(:,3),y1(:,2),'r*','Markersize',20);ylabel('条件数','fontsize',20);xlabel('最大扭矩','fontsize',20);set(gca,'fontsize',20);
%}