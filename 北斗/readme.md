2020上海北斗高分比赛  参考2020山西大同照激光比赛

# yolov3_tiny + movidius + Ros melodic18.04 + upcore + realsense


# 一.安装环境

## 1.安装realsense
[参考realsense](https://github.com/IntelRealSense/realsense-ros)
#### Step1.
`sudo apt-get install ros-melodic-realsense2-camera`

## 2.安装Openvino
[参考openvino](https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick)
#### Step1.下载openvino安装包
最好直接下载我使用的版本[2020.4.287](https://pan.baidu.com/s/1X1k8_Hwbyhu7Na1Nx0-WXg) 提取码：jhbo 
#### Step2.解压安装包
`tar xvf l_openvino_toolkit_p_2020.4.287.tgz`
#### Step3.进入安装包
`cd l_openvino_toolkit_p_2020.4.287`
#### Step4.安装依赖
`sudo -E ./install_openvino_dependencies.sh`
#### Step5.运行安装GUI(建议添加sudo，添加sudo命令安装到根目录下，不加sudo命令默认安装到home目录下)
`sudo ./install_GUI.sh`
#### Step6.安装完成之后添加source到bashrc下
```
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
source ~/.bashrc
```

## 3.安装movidius(NCS2)依赖
`cd ~/intel/openvino/install_dependencies`  
`sudo ./install_NCS_udev_rules.sh`

## 4.配置Model Optimizer的依赖（此项较久upcore需要20分钟）
[参考OpenvinoToolkit](https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux.html#install-external-dependencies)    
`cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites`
`sudo ./install_prerequisites.sh`

## 5.检查安装情况
#### Step1.命令行进入python3
`python3`
#### Step2.运行以下命令看是否有报错
```
import cv2
import openvino
import tensorflow
```
#### Step3.报错(如果import没有报错则可以跳过Step3)
如果`import tensorflow`报错core dumped：
![image.png](https://i.loli.net/2020/10/31/d8ALNvUqgIHP1ci.png)
这是因为upcore的CPU版本太老，不支持tensorflow1.6.0以后的AVX指令集

#### Step3.解决方案
重装tensorflow 1.5.0版本，安装完成后重新运行Step2的指令，确保能够成功import cv2 openvino tensorflow    
```
sudo pip3 uninstall tensorflow
sudo pip3 install tensorflow==1.5.0
```




# 二.测试识别demo

