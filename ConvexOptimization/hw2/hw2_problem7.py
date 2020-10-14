#Problem 7
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

## plot
x = np.linspace(1,2,2000)
y = x*x+4*np.cos(x)
fig1 = plt.figure(1)
plt.plot(x,y)
plt.draw()
plt.pause(2)
plt.close(fig1)


## golden section
def func(x):
    return x*x+4*np.cos(x)

def compare(a,b,left,right):
    if(func(left)>func(right)):
        return '[{},{}]'.format(left,b)
    else:
        return  '[{},{}]'.format(a,right)

count = 1
a = 1
b = 2
uncertainty = 0.2
left = a + 0.382*(b-a)
right = a + 0.618*(b-a)

########################### table ######################
table = PrettyTable(['Iteration k','ak','bk','f(ak)','f(bk)','New uncertainty interval'])
table.add_row(['0','1','2',func(1),func(2),'[1,2]'])

while(not b-a < uncertainty):
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


## Newton's 1st method

def funcnew(x):
    return x*x+4*np.cos(x)

def direvitive(x):
    return 2*x - 4*np.sin(x)
table2 = PrettyTable(['Iteration k','xk(Newton1st)','f(xk)'])
table2.add_row(['0','1',funcnew(1)])
x0 = 1
for i in range(0,count):
    x1 = x0 - 0.1*funcnew(x0)/direvitive(x0)
    table2.add_row([i+1,x1,funcnew(x1)])
    x0 = x1

print(table2)



## Newton's 2nd method
def direvitive2(x):
    return 2 - 4*np.cos(x)
table3 = PrettyTable(['Iteration k','xk(Newton2nd)','f(xk)'])
table3.add_row(['0','1',funcnew(1)])
x0 = 1
for i in range(0,count):
    x1 = x0 - direvitive(x0)/direvitive2(x0)
    table3.add_row([i+1,x1,funcnew(x1)])
    x0 = x1

print(table3)
