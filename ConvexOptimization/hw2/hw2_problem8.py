# secant method
from prettytable import PrettyTable
import numpy as np

def f(x):
    return (2*x-1)**2 + 4*(4-1024*x)**4

def direvitive(x):
    return 4*x+16*(4-1024*x)**3
x0 = 0
x1 = 1
f0 = f(x0)
f1 = f(x1)
count = 0
table = PrettyTable(['Iteration k','x0','x1'])
table.add_row(['Iteration {}'.format(count),'x{}={}'.format(count,x0),'x{}={}'.format(count+1,x1)])
div0 = direvitive(x0)
while(np.abs(x1-x0)>=np.abs(x0)*1e-5):
    div1 = direvitive(x1)
    x2 = x1 - div1*(x1-x0)/(div1-div0)
    x0,x1 = x1,x2
    div0 = div1
    count = count + 1
    table.add_row(['Iteration {}'.format(count),'x{}={}'.format(count,x0),'x{}={}'.format(count+1,x1)])
print(table)
print('Last x = {},f(x)={}'.format(x1,f(x1)))
