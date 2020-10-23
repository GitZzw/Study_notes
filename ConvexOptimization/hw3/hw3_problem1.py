#Problem 1 HW3
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

## golden section
def func(x):
    temp = np.mat(np.array([[2,1],[1,2]]))
    return 1/2*(x)*temp*(x.T)

def compare(a,b,left,right):
    if(func(left)>func(right)):
        return '[{},{}]'.format(left,b)
    else:
        return  '[{},{}]'.format(a,right)

count = 1
a = np.mat(np.array([0.5975,-0.2950]))
b = np.mat(np.array([-0.0100,0.4300]))
uncertainty = 0.01
left = a + 0.382*(b-a)
right = a + 0.618*(b-a)

########################### table ######################
table = PrettyTable(['Iteration k','ak','bk','f(ak)','f(bk)','New uncertainty interval'])

table.add_row(['0','1','2',func(a),func(b),'[a0,b0]'])

while(count < 10):
    table.add_row([count,left,right,func(left),func(right),compare(a,b,left,right)])
    if(func(left)>func(right)):
        a = left
        b = b
        left = right
        right = a + 0.618*(b-a)
    else:
        a = a
        b = right
        right = left
        left = a + 0.382*(b-a)
    count = count + 1

print(table)
print('interval is:  ',0.008201513)
