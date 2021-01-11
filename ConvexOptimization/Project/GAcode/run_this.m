clear
clc
fitnessfcn=@linf;   %优化函数句柄
nonlcon=@nonf3;
nvars=11;        %变量个数
lb=[0,0,0,0,0,0,0,0,0,0,0];      %下限
ub=[90,90,180,180,180,180,180,180,360,360,360];     %上限
A1=[zeros(1,8),1,-1,0;zeros(1,8),0,1,-1;zeros(1,8),-1,0,1];b=[-40;-40;320];     %线性不等式约束
Aeq=[];beq=[];  %线性等式约束
options=gaoptimset('paretoFraction',0.4,'populationsize',100,'generations',300,'stallGenLimit',300,'TolFun',1e-100,...
'CrossoverFraction',0.95,'MigrationFraction',0.9,'PlotFcns',{@gaplotpareto});
% 最优个体系数paretoFraction为0.4；种群大小populationsize为100，最大进化代数generations为300，
% 停止代数stallGenLimit为300， 适应度函数偏差TolFun设为1e-100，函数gaplotpareto：绘制Pareto前端
[x,fval]=gamultiobj(fitnessfcn,nvars,A1,b,Aeq,beq,lb,ub,options);

