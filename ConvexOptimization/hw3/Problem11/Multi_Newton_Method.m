function [x_optimization,f_optimization] = Multi_Newton_Method(f,x0,var_x,epsilon)
format long;
%   f：目标函数
%   x0：初始点
%   var_x：自变量向量
%   epsilon：精度
%   x_optimization：目标函数取最小值时的自变量值
%   f_optimization：目标函数的最小值
if nargin == 3
    epsilon = 1.0e-3;
end
x0 = transpose(x0);
var_x = transpose(var_x);
gradf = jacobian(f,var_x);
grad2f = jacobian(gradf,var_x);
grad_fxk = 1;
k = 0;
xk = x0;
xstore = [norm(xk)];
fxstore = [(xk(1)+10*xk(2))^2+5*(xk(3)-xk(4))^2+(xk(2)-2*xk(3))^4+10*(xk(1)-xk(4))^4];
count = 0;
while norm(grad_fxk) >= epsilon*max(1,norm(xk)) % stop criteria
    grad_fxk  = subs(gradf,var_x,xk);
    grad2_fxk = subs(grad2f,var_x,xk);
    pk = -inv(grad2_fxk)*transpose(grad_fxk);  
    pk = double(pk);
    xk_next = xk + pk; 
    xk = xk_next;
    k = k + 1;
    xstore = [xstore,norm(xk)];
    fxstore = [fxstore,(xk(1)+10*xk(2))^2+5*(xk(3)-xk(4))^2+(xk(2)-2*xk(3))^4+10*(xk(1)-xk(4))^4];
    count = count + 1;
end
figure;
plot([0:1:count],xstore)
title('||xk-0||')
figure;
plot([0:1:count],fxstore)
title('f(xk)')
x_optimization = xk_next;
f_optimization = fxstore(length(fxstore)-1);
format short;
