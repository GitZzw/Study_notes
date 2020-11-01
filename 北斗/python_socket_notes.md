 > For linux  
 sudo apt-get update出现W: GPG 错误：http://packages.ros.org/ros/ubuntu xenial InRelease: 由于没有公钥，无法验证下列签名：
 ```
 W: GPG 错误：http://packages.ros.org/ros/ubuntu xenial InRelease: 由于没有公钥，无法验证下列签名： NO_PUBKEY F42ED6FBAB17C654
W: 仓库 “http://packages.ros.org/ros/ubuntu xenial InRelease” 没有数字签名。
N: 无法认证来自该源的数据，所以使用它会带来潜在风险。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节。
```
解决
` sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654 ` 注意：最后的码，根据不同的电脑进行修改










# python socket.error: [Errno 48] Address already in use

## 1.在ctrl+c结束服务器端程序后，再次启动运行程序会出现Address already in use这个错误，过几分钟运行或重启后运行又好了，那是因为操作系统会在服务器socket被关闭或服务器进程终止后会将该端口保留几分钟，而要解决该问题，可以在socket定义之后加上下面这句话：
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 这里s为定义的socket,这样操作系统会在服务器socket被关闭或服务器进程终止后马上释放该服务器的端口，下次运行就不会出现上述问题啦。

## 2.使用lsof命令
打开终端，输入sudo lsof -i:8000
8000是对应的端口号
然后执行kill -9 3428(PID)

## 3.打开终端，输入ps，回车
找到含有对应节点关键词的PID，kill
ps -ef |grep “python”

## 4.socket.close()的使用
填坑
