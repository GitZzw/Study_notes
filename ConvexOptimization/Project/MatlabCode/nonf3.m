function [c,ceq]=nonf3(x)
%x=[75.0815126800000,55.5752325900000,83.5011604400000,104.925344500000,59.1447495000000,92.8806014200000,84.4107349600000,92.2283726300000,109.363643900000,17.0468128900000,63.6683774200000,117.785187500000]
ceq=[];
a1=-sind(x(1,4:6)).*cosd(x(1,10:12));
a2=-sind(x(1,4:6)).*sind(x(1,10:12)).*cosd(x(2));
a3=-cosd(x(1,4:6)).*sind(x(1,10:12)).*sind(x(2));
a4=cosd(x(1,4:6)).*cosd(x(1,10:12)).*sind(x(2));
a5=-sind(x(1,4:6)).*sind(x(1,10:12));
a6=sind(x(1,4:6)).*cosd(x(1,10:12)).*cosd(x(2));
a7=-cosd(x(1,4:6)).*cosd(x(2));
a8=sind(x(1,4:6)).*sind(x(2));

%初始值
wc1=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*[0;sind(x(3));cosd(x(3))];
wc2=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*[0;sind(x(3));cosd(x(3))];
wc3=[cosd(x(12)),-sind(x(12)),0;sind(x(12)),cosd(x(12)),0;0,0,1]*[0;sind(x(3));cosd(x(3))];


wc=[wc1,wc2,wc3];
u1=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*[0;1;0];

u2=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*[0;1;0];

u3=[cosd(x(12)),-sind(x(12)),0;sind(x(12)),cosd(x(12)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*[0;1;0];
%{
%求初始位置
w=wc;%i=1为第一列
A=w(1,:).*(-a1+a3)+w(2,:).*(a4-a5)+w(3,:).*a7-cosd(x(1,7:9));%ABCD行向量
B=w(1,:).*2.*a2+w(2,:).*2.*a6+w(3,:).*2.*a8;
C=w(1,:).*(a1+a3)+w(2,:).*(a4+a5)+w(3,:).*a7-cosd(x(1,7:9));
D=B.^2-4.*A.*C;

if (D(1,1)>0)&(D(1,2)>0)&(D(1,3)>0)
if A(1,1)==0
    thetac1=2*atand(-C(1,1)/B(1,1));
else
    thetac1=2*atand((-B(1,1)+sqrt(D(1,1)))./2./A(1,1));
end
if A(1,2)==0
    thetac2=2*atand(-C(1,2)/B(1,2));
else
    thetac2=2*atand((-B(1,2)+sqrt(D(1,2)))./2./A(1,2));
end
if A(1,3)==0
    thetac3=2*atand(-C(1,3)/B(1,3));
else
    thetac3=2*atand((-B(1,3)+sqrt(D(1,3)))./2./A(1,3));
end

vc1=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*...
[cosd(thetac1),0,sind(thetac1);0,1,0;-sind(thetac1),0,cosd(thetac1)]*...
[cosd(x(4)),-sind(x(4)),0;sind(x(4)),cosd(x(4)),0;0,0,1]*[0;1;0];

vc2=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*...
[cosd(thetac2),0,sind(thetac2);0,1,0;-sind(thetac2),0,cosd(thetac2)]*...
[cosd(x(5)),-sind(x(5)),0;sind(x(5)),cosd(x(5)),0;0,0,1]*[0;1;0];

vc3=[cosd(x(12)),-sind(x(12)),0;sind(x(12)),cosd(x(12)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*...
[cosd(thetac3),0,sind(thetac3);0,1,0;-sind(thetac3),0,cosd(thetac3)]*...
[cosd(x(6)),-sind(x(6)),0;sind(x(6)),cosd(x(6)),0;0,0,1]*[0;1;0];

        %统一分支约束
        lin1=sign(dot(cross(vc1,u1),wc1));
        lin2=sign(dot(cross(vc2,u2),wc2));
        lin3=sign(dot(cross(vc3,u3),wc3));
        %主动杆干涉
        lin4=sign(dot(cross(vc2,u2),vc1));
        lin5=sign(dot(cross(vc3,u3),vc2));
        lin6=sign(dot(cross(vc1,u1),vc3));
        %从动杆干涉
        lin7=sign(dot(cross(vc2,wc2),vc1));
        lin8=sign(dot(cross(vc3,wc3),vc2));
        lin9=sign(dot(cross(vc1,wc1),vc3));
else
  lin1=0;lin2=0;lin3=0;lin4=0;lin5=0;lin6=0;lin7=0;lin8=0;lin9=0;
  
end
%}

%%
%循环结构

c=-ones(1,1320*9);

count=1;
for afa=-40:10:50
for beta=-10:10:90
for gama=-90:10:20    
     
Rz=[cosd(x(1)),-sind(x(1)),0;sind(x(1)),cosd(x(1)),0;0,0,1];  
RPY=[cosd(afa),-sind(afa),0;sind(afa),cosd(afa),0;0,0,1]*...
    [cosd(beta),0,sind(beta);0,1,0;-sind(beta),0,cosd(beta)]*...
    [1,0,0;0,cosd(gama),-sind(gama);0,sind(gama),cosd(gama)];
Q=Rz*RPY;
w=Q*wc;%i=1为第一列
A=w(1,:).*(-a1+a3)+w(2,:).*(a4-a5)+w(3,:).*a7-cosd(x(1,7:9));%ABCD行向量
B=w(1,:).*2.*a2+w(2,:).*2.*a6+w(3,:).*2.*a8;
C=w(1,:).*(a1+a3)+w(2,:).*(a4+a5)+w(3,:).*a7-cosd(x(1,7:9));
D=B.^2-4.*A.*C;

if (D(1,1)>0)&(D(1,2)>0)&(D(1,3)>0)

if A(1,1)==0
    thetac1=2*atand(-C(1,1)/B(1,1));
else
    thetac1=2*atand((-B(1,1)+sqrt(D(1,1)))./2./A(1,1));
end

if A(1,2)==0
    thetac2=2*atand(-C(1,2)/B(1,2));
else
    thetac2=2*atand((-B(1,2)+sqrt(D(1,2)))./2./A(1,2));
    
end

if A(1,3)==0
    thetac3=2*atand(-C(1,3)/B(1,3));
else
    thetac3=2*atand((-B(1,3)-sqrt(D(1,3)))./2./A(1,3));
    
end
    
v1=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*...
[cosd(thetac1),0,sind(thetac1);0,1,0;-sind(thetac1),0,cosd(thetac1)]*...
[cosd(x(4)),-sind(x(4)),0;sind(x(4)),cosd(x(4)),0;0,0,1]*[0;1;0];

v2=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*...
[cosd(thetac2),0,sind(thetac2);0,1,0;-sind(thetac2),0,cosd(thetac2)]*...
[cosd(x(5)),-sind(x(5)),0;sind(x(5)),cosd(x(5)),0;0,0,1]*[0;1;0];

v3=[cosd(x(12)),-sind(x(12)),0;sind(x(12)),cosd(x(12)),0;0,0,1]*...
[1,0,0;0,cosd(x(2)-90),-sind(x(2)-90);0,sind(x(2)-90),cosd(x(2)-90)]*...
[cosd(thetac3),0,sind(thetac3);0,1,0;-sind(thetac3),0,cosd(thetac3)]*...
[cosd(x(6)),-sind(x(6)),0;sind(x(6)),cosd(x(6)),0;0,0,1]*[0;1;0];
    w1=w(:,1);
    w2=w(:,2);
    w3=w(:,3);
    %
    if count==1
        %统一分支约束
        lin1=sign(dot(cross(v1,u1),w1));
        lin2=sign(dot(cross(v2,u2),w2));
        lin3=sign(dot(cross(v3,u3),w3));
        %{
        %主动杆干涉
        lin4=sign(dot(cross(v2,u2),v1));
        lin5=sign(dot(cross(v3,u3),v2));
        lin6=sign(dot(cross(v1,u1),v3));
        %从动杆干涉
        lin7=sign(dot(cross(v2,w2),v1));
        lin8=sign(dot(cross(v3,w3),v2));
        lin9=sign(dot(cross(v1,w1),v3));
%}
    end
    
%统一分支约束
c(1,count)=0.1-sign(dot(cross(v1,u1),w1))*lin1;
c(1,count+1)=0.1-sign(dot(cross(v2,u2),w2))*lin2;
c(1,count+2)=0.1-sign(dot(cross(v3,u3),w3))*lin3;
%{

%主动杆干涉
c(1,count+3)=0.1-sign(dot(cross(v2,u2),v1))*lin4;
c(1,count+4)=0.1-sign(dot(cross(v3,u3),v2))*lin5;
c(1,count+5)=0.1-sign(dot(cross(v1,u1),v3))*lin6;
 %从动杆干涉
c(1,count+6)=0.1-sign(dot(cross(v2,w2),v1))*lin7;
c(1,count+7)=0.1-sign(dot(cross(v3,w3),v2))*lin8;
c(1,count+8)=0.1-sign(dot(cross(v1,w1),v3))*lin9;
%}
count=count+9;
end



end
end
end


end