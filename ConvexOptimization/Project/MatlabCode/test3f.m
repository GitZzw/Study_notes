% https://blog.csdn.net/sinat_34054843/article/details/76451504

% https://blog.csdn.net/Weizhiyuan37927/article/details/77622136
clear
clc
fitnessfcn=@linf;   %��Ӧ�Ⱥ������
nonlcon=@nonf3;
nvars=11;        %��������
lb=[0,0,0,0,0,0,0,0,0,0,0];      %����
ub=[90,90,180,180,180,180,180,180,360,360,360];     %����
A1=[zeros(1,8),1,-1,0;zeros(1,8),0,1,-1;zeros(1,8),-1,0,1];b=[-40;-40;320];     %���Բ���ʽԼ��
Aeq=[];beq=[];  %���Ե�ʽԼ��
options=gaoptimset('paretoFraction',0.4,'populationsize',100,'generations',300,'stallGenLimit',300,'TolFun',1e-100,...
'CrossoverFraction',0.95,'MigrationFraction',0.9,'PlotFcns',{@gaplotpareto});
% ���Ÿ���ϵ��paretoFractionΪ0.3����Ⱥ��СpopulationsizeΪ100������������generationsΪ200��
% ֹͣ����stallGenLimitΪ200�� ��Ӧ�Ⱥ���ƫ��TolFun��Ϊ1e-100������gaplotpareto������Paretoǰ��
%[x,fval]=gamultiobj(fitnessfcn,nvars,A1,b,Aeq,beq,lb,ub,nonlcon,options);
[x,fval]=gamultiobj(fitnessfcn,nvars,A1,b,Aeq,beq,lb,ub,options);



%plot3(fval(:,1),fval(:,2),fval(:,3),'*');
%plot(fval(:,2),fval(:,3),'*')
%{
x1=find(fval(:,1) == -23)
y1=fval(x1,:);
xx=x(x1,:);
plot(y1(:,3),y1(:,2),'r*','Markersize',20);ylabel('������','fontsize',20);xlabel('���Ť��','fontsize',20);set(gca,'fontsize',20);
%}