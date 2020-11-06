function [x,N]=steep_desc(grad,xnew,options)

if nargin ~= 3
    options = [];
    if nargin ~= 2
        disp('Wrong number of arguments.');
        return;
    end
end
if length(options) >= 14
    if options(14)==0
        options(14)=1000*length(xnew);
    end
else
    options(14)=1000*length(xnew);
end

format compact;
format short e;
options = foptions(options);
print = options(1);
epsilon_x = options(2);
epsilon_g = options(3);
max_iter=options(14);
for k = 1:max_iter
    xcurr=xnew;
    g_curr=feval(grad,xcurr);
    if norm(g_curr) <= epsilon_g
        disp('Terminating: Norm of gradient less than');
        disp(epsilon_g);
        k=k-1;
        break;
    end %if
    alpha=linesearch_secant(grad,xcurr,-g_curr);
    xnew = xcurr-alpha*g_curr;
    if print
        disp('Iteration number k =')
        disp(k); %print iteration index k
        disp('alpha =');
        disp(alpha); %print alpha
        disp('Gradient = ');
        disp(g_curr'); %print gradient
        disp('New point =');
        disp(xnew'); %print new point
    end %if
    if norm(xnew-xcurr) <= epsilon_x*norm(xcurr)
        disp('Terminating: Norm of difference between iterates less than');
        disp(epsilon_x);
        break;
    end %if
    if k == max_iter
    disp('Terminating with maximum number of iterations');
    end %if
end %for
if nargout >= 1
    x=xnew;
if nargout == 2
    N=k;
end
else
    disp('Final point =');
    disp(xnew');
    disp('Number of iterations =');
    disp(k);
end %if
