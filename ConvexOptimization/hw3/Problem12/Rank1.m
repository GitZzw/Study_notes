function [x_optimization,f_optimization] = Modified_Multi_Newton_Method(f,x0,var_x,epsilon)
format long;
%   f：目标函数
%   x0：初始点
%   var_x：自变量向量
%   epsilon：精度
%   x_optimization：目标函数取最小值时的自变量值
%   f_optimization：目标函数的最小值
if nargin == 3
    epsilon = 1.0e-6;
end
x0 = transpose(x0);
var_x = transpose(var_x);
syms t;
gradf = jacobian(f,var_x);
grad2f = jacobian(gradf,var_x);
grad_fxk = 1;
k = 0;
xk = x0;

while norm(grad_fxk) > epsilon
    grad_fxk  = subs(gradf,var_x,xk);
    grad2_fxk = subs(grad2f,var_x,xk);
    pk = -inv(grad2_fxk)*transpose(grad_fxk);
    yk = xk + t*pk;
    fyk = subs(f,var_x,yk);
    [xmin,xmax] = Advance_and_Retreat_Method(fyk,0,0.1);
    tk = Golden_Section_Method(fyk,xmin,xmax);
    xk_next = xk + tk*pk;
    xk = xk_next;
    k = k + 1;
end

x_optimization = xk_next;
f_optimization = subs(f,var_x,x_optimization);
format short;