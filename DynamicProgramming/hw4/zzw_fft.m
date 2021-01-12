function F=zzw_fft(x)
%% 判断是否是向量
sz=size(x);
if sz(1)>1&&sz(2)>1||numel(sz)>2
    F=-1;
    return;
end
%% 
N=max(sz);
if N==1     
    F=x;
    return;
end
if sz(1)>1
    F=zeros(N,1);
end
if sz(2)>1
    F=zeros(1,N);
end
for k=1:N/2
    [F(k),F(k+N/2)]=calculate_element(x,N,k);
end


function [Fk,Fkn]=calculate_element(x,N,k) %输入信号x,信号总长度N，频域坐标k
    if N==1
        Fk=x;
        Fkn=x;
        return;
    else
        x1=x(1:2:N-1);%奇数
        x2=x(2:2:N);%偶数
        F1=calculate_element(x1,N/2,k);
        F2=calculate_element(x2,N/2,k);
        Wkn=exp(-i*2*pi*(k-1)/N);
        Fk=F1+Wkn*F2;
        Fkn=F1-Wkn*F2;
    end
