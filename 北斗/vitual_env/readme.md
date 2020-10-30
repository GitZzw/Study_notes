# 1.`sudo apt-get install python3`

# 2.`sudo apt-get install python3=pip`

# 3.`sudo apt install vitualenv`

# 4.`vitualenv .zzw --python=python3`

# 5.`source .zzw/bin/activate`

# 6.`pip3 install opencv-python`

# 7.查看python包`pip3 list`

# 8.python3 import tensorflow 显示illegal instruction core dumped，可能是因为cpu太老不支持新版tensorflow（pip3 list 查看版本）的AVX指令集
> 解决：
`
>>sudo pip3 uninstall tensorflow
>>sudo pip3 install tensorflow==1.5.0
`

# 9.python pip 和 pip3 的差别
> 先搜索了一下看到了如下的解释，安装了python3之后，库里面既会有pip3也会有pip
使用pip install XXX新安装的库会放在这个目录下面
python2.7/site-packages
使用pip3 install XXX，新安装的库会放在这个目录下面
python3.6/site-packages
如果使用python3执行程序，那么就不能import python2.7/site-packages中的库
