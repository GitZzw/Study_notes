function x=DFP(f,x0,eps,k)
%目标函数f
%初始迭代点x0
%迭代精度eps
%迭代次数K
x0=x0';
TiDu=matlabFunction(gradient(sym(f)));
m=length(x0);
H=eye(m);%构造初始海塞矩阵
%写出函数值，梯度值的表达式
f_cal='f(x0(1),x0(2))'; %初始点函数值
tidu_cal='TiDu(x0(1),x0(2))';
f1_cal='f(x_1(1),x_1(2))';%下一点函数值
%% 从第一个点计算到第二个点
f_x0=eval(f_cal);%计算初始点函数值
tidu_x0=eval(tidu_cal);%计算初始点梯度值
if norm(tidu_x0) < eps%判断是否满足终止条件
    x=x0;
    return;
end
d=-A*tidu_x0;% 下降方向
syms alfa %定义步长
x_1=x0+alfa*d;%更新迭代点位置
f_x1=eval(f1_cal);%计算迭代点的函数表达式
df_x1=diff(f_x1);%迭代点的梯度表达式
dalfa=double(solve(df_x1));%求解，得到步长alfa
x0=x0+dalfa*d;%更新初始点
tidu_x1=eval(tidu_cal);%计算该点梯度
n=1;
while n < k && norm(tidu_x1) > eps
    delta_x=dalfa*d;  %计算sK
    delta_g=tidu_x1-tidu_x0; %计算yk
    delta_H=delta_x*delta_x'/(delta_x'*delta_g)- H*delta_g*delta_g'*H/(delta_g'*H*delta_g); %计算delta_Dk
    H=H+delta_H;   %dfp迭代公式
    tidu_x0=tidu_x1; %将该点梯度作为新的初始点继续迭代
    
    d=-A*tidu_x0;% 下降方向
    syms alfa
    x_1=x0+alfa*d;
    f_x1=eval(f1_cal);
    df_x1=diff(f_x1);
    dalfa=double(solve(df_x1));
    x0=x0+dalfa*d;
    tidu_x1=eval(tidu_cal);
end
x=x0;
end