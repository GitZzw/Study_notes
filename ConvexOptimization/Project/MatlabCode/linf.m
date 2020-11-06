function y=linf(x)

a1=-sind(x(1,3:5)).*cosd(x(1,9:11));
a2=-sind(x(1,3:5)).*sind(x(1,9:11)).*cosd(x(1));
a3=-cosd(x(1,3:5)).*sind(x(1,9:11)).*sind(x(1));
a4=cosd(x(1,3:5)).*cosd(x(1,9:11)).*sind(x(1));
a5=-sind(x(1,3:5)).*sind(x(1,9:11));
a6=sind(x(1,3:5)).*cosd(x(1,9:11)).*cosd(x(1));
a7=-cosd(x(1,3:5)).*cosd(x(1));
a8=sind(x(1,3:5)).*sind(x(1));
wc1=[cosd(x(9)),-sind(x(9)),0;sind(x(9)),cosd(x(9)),0;0,0,1]*[0;sind(x(2));cosd(x(2))];
wc2=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*[0;sind(x(2));cosd(x(2))];
wc3=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*[0;sind(x(2));cosd(x(2))];

wc=[wc1,wc2,wc3];
u1=[cosd(x(9)),-sind(x(9)),0;sind(x(9)),cosd(x(9)),0;0,0,1]*...
[1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*[0;1;0];

u2=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
[1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*[0;1;0];

u3=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*[0;1;0];
%%
%循环结构

countw=0;
countk10=0;
countt=0;
F=[10.5;0;0];
flag=0; 

for afa=-40:10:50
for beta=-10:10:90
for gama=-90:10:20   
    
    
Q=[cosd(afa),-sind(afa),0;sind(afa),cosd(afa),0;0,0,1]*...
    [cosd(beta),0,sind(beta);0,1,0;-sind(beta),0,cosd(beta)]*...
    [1,0,0;0,cosd(gama),-sind(gama);0,sind(gama),cosd(gama)];
w=Q*wc;%i=1为第一列
    w1=w(:,1);
    w2=w(:,2);
    w3=w(:,3);
A=w(1,:).*(-a1+a3)+w(2,:).*(a4-a5)+w(3,:).*a7-cosd(x(1,6:8));%ABCD行向量
B=w(1,:).*2.*a2+w(2,:).*2.*a6+w(3,:).*2.*a8;
C=w(1,:).*(a1+a3)+w(2,:).*(a4+a5)+w(3,:).*a7-cosd(x(1,6:8));
D=B.^2-4.*A.*C;

if (D(1,1)>0)&(D(1,2)>0)&(D(1,3)>0)
countw=countw+1;

if A(1,1)==0
    thetac1=2*atand(-C(1,1)/B(1,1));
    v1=[cosd(x(9)),-sind(x(9)),0;sind(x(9)),cosd(x(9)),0;0,0,1]*...
    [1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
    [cosd(thetac1),0,sind(thetac1);0,1,0;-sind(thetac1),0,cosd(thetac1)]*...
    [cosd(x(3)),-sind(x(3)),0;sind(x(3)),cosd(x(3)),0;0,0,1]*[0;1;0];

else
    thetac1a=2*atand((-B(1,1)+sqrt(D(1,1)))./2./A(1,1));
    v1a=[cosd(x(9)),-sind(x(9)),0;sind(x(9)),cosd(x(9)),0;0,0,1]*...
    [1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
    [cosd(thetac1a),0,sind(thetac1a);0,1,0;-sind(thetac1a),0,cosd(thetac1a)]*...
    [cosd(x(3)),-sind(x(3)),0;sind(x(3)),cosd(x(3)),0;0,0,1]*[0;1;0];
 
    thetac1b=2*atand((-B(1,1)-sqrt(D(1,1)))./2./A(1,1));
    v1b=[cosd(x(9)),-sind(x(9)),0;sind(x(9)),cosd(x(9)),0;0,0,1]*...
    [1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
    [cosd(thetac1b),0,sind(thetac1b);0,1,0;-sind(thetac1b),0,cosd(thetac1b)]*...
    [cosd(x(3)),-sind(x(3)),0;sind(x(3)),cosd(x(3)),0;0,0,1]*[0;1;0];

   if(dot(cross(u1,w1),v1a)<-0.01)
       v1=v1a;
   else if(dot(cross(u1,w1),v1b)<-0.01) 
           v1=v1b;
       else
           v1=v1a;
          flag=1;
       end
    end
end

if A(1,2)==0
    thetac2=2*atand(-C(1,2)/B(1,2));
    v2=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
    [1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
    [cosd(thetac2),0,sind(thetac2);0,1,0;-sind(thetac2),0,cosd(thetac2)]*...
    [cosd(x(4)),-sind(x(4)),0;sind(x(4)),cosd(x(4)),0;0,0,1]*[0;1;0];
else
    thetac2a=2*atand((-B(1,2)+sqrt(D(1,2)))./2./A(1,2));
    v2a=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
    [1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
    [cosd(thetac2a),0,sind(thetac2a);0,1,0;-sind(thetac2a),0,cosd(thetac2a)]*...
    [cosd(x(4)),-sind(x(4)),0;sind(x(4)),cosd(x(4)),0;0,0,1]*[0;1;0];

    thetac2b=2*atand((-B(1,2)-sqrt(D(1,2)))./2./A(1,2));
    v2b=[cosd(x(10)),-sind(x(10)),0;sind(x(10)),cosd(x(10)),0;0,0,1]*...
    [1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
    [cosd(thetac2b),0,sind(thetac2b);0,1,0;-sind(thetac2b),0,cosd(thetac2b)]*...
    [cosd(x(4)),-sind(x(4)),0;sind(x(4)),cosd(x(4)),0;0,0,1]*[0;1;0];

     if(dot(cross(u2,w2),v2a)<-0.01)
       v2=v2a;
   else if(dot(cross(u2,w2),v2b)<-0.01) 
           v2=v2b;
       else
           v2=v2a;
          flag=1;
       end
    end
    
end

if A(1,3)==0
    thetac3=2*atand(-C(1,3)/B(1,3));
    v3=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
[cosd(thetac3),0,sind(thetac3);0,1,0;-sind(thetac3),0,cosd(thetac3)]*...
[cosd(x(5)),-sind(x(5)),0;sind(x(5)),cosd(x(5)),0;0,0,1]*[0;1;0];
else
    thetac3a=2*atand((-B(1,3)+sqrt(D(1,3)))./2./A(1,3));
    v3a=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
[cosd(thetac3a),0,sind(thetac3a);0,1,0;-sind(thetac3a),0,cosd(thetac3a)]*...
[cosd(x(5)),-sind(x(5)),0;sind(x(5)),cosd(x(5)),0;0,0,1]*[0;1;0];

    thetac3b=2*atand((-B(1,3)-sqrt(D(1,3)))./2./A(1,3));
    v3b=[cosd(x(11)),-sind(x(11)),0;sind(x(11)),cosd(x(11)),0;0,0,1]*...
[1,0,0;0,cosd(x(1)-90),-sind(x(1)-90);0,sind(x(1)-90),cosd(x(1)-90)]*...
[cosd(thetac3b),0,sind(thetac3b);0,1,0;-sind(thetac3b),0,cosd(thetac3b)]*...
[cosd(x(5)),-sind(x(5)),0;sind(x(5)),cosd(x(5)),0;0,0,1]*[0;1;0];

 if(dot(cross(u3,w3),v3a)<-0.01)
       v3=v3a;
   else if(dot(cross(u3,w3),v3b)<-0.01) 
           v3=v3b;
       else
           v3=v3a;
          flag=1;
       end
    end
   
end
    

Jx=[cross(v1,w1),cross(v2,w2),cross(v3,w3)]';
Kq=diag([dot(cross(v1,u1),w1),dot(cross(v2,u2),w2),dot(cross(v3,u3),w3)],0);
J=-inv(Kq)*Jx;%运动学反解雅克比  
k=norm(J,2)*norm(inv(J),2);
if k<10
countk10=countk10+1;    
end

t=inv(J')*F;
maxt=max(t);
if maxt>countt
    countt=maxt;
end

end



end
end
end
%%




%
if countw<1000%防止工作空间太小
    y(1)=inf;
    y(2)=inf;
    y(3)=inf;
else
%}
y(1)=-countw;
y(3)=countt;

if flag==0
y(2)=-countk10/countw;
else
   y(2)=inf;
end


end





