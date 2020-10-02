## 基于深度学习框架darknet的yolov4算法应用
### [参考资料]:(https://github.com/AlexeyAB/darknet)

#### 1.下载darknet包

> \>>git clone https://github.com/AlexeyAB/darknet.git


#### 2.下载两个文件yolov4run.py和configer.py到/darknet目录下

> 下载地址:(https://github.com/GitZzw/Study_notes/tree/master/YOLOv4)


#### 3.在/darknet/data文件夹下创建pic txt xml三个文件夹


#### 4.将训练图像存放到/darknet/data/pic文件夹下,将对应图像名的对应xml文件存放到/darknet/data/xml文件夹下

> xml文件获得方法使用 **labelImg** 工具(https://github.com/tzutalin/labelImg)


#### 5.运行yolov4run.py,然后根据提示操作即可
> \>>python yolov4run.py
