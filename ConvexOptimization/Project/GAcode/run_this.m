clear
clc
fitnessfcn=@linf;   %�Ż��������
nonlcon=@nonf3;
nvars=11;        %��������
lb=[0,0,0,0,0,0,0,0,0,0,0];      %����
ub=[90,90,180,180,180,180,180,180,360,360,360];     %����
A1=[zeros(1,8),1,-1,0;zeros(1,8),0,1,-1;zeros(1,8),-1,0,1];b=[-40;-40;320];     %���Բ���ʽԼ��
Aeq=[];beq=[];  %���Ե�ʽԼ��
options=gaoptimset('paretoFraction',0.4,'populationsize',100,'generations',300,'stallGenLimit',300,'TolFun',1e-100,...
'CrossoverFraction',0.95,'MigrationFraction',0.9,'PlotFcns',{@gaplotpareto});
% ���Ÿ���ϵ��paretoFractionΪ0.4����Ⱥ��СpopulationsizeΪ100������������generationsΪ300��
% ֹͣ����stallGenLimitΪ300�� ��Ӧ�Ⱥ���ƫ��TolFun��Ϊ1e-100������gaplotpareto������Paretoǰ��
[x,fval]=gamultiobj(fitnessfcn,nvars,A1,b,Aeq,beq,lb,ub,options);

