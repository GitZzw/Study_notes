A=[1 0 1 0 0; 0 1 0 1 0; 1 1 0 0 1];
b=[4;6;8];
c=[-2;-5;0;0;0];
v=[3;4;5];
options(1)=1;
[x,v]=simplex(c,A,b,v,options);
