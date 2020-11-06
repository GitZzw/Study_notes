syms x1 x2 x3 x4;
f = (x1+10*x2)^2+5*(x3-x4)^2+(x2-2*x3)^4+10*(x1-x4)^4;
[x_optimization,f_optimization] = Multi_Newton_Method(f,[3 -1 0 1],[x1 x2 x3 x4]);
x_optimization
f_optimization