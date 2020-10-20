## Installation Instructions

### 1.安装openvino+realsense

> 参考   
  openvino: [https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick](https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick)   
  realsense：[https://github.com/IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros)  
  
  
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
