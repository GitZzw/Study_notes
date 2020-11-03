#!/usr/bin/python
# coding: utf-8

import socket
from time import time
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3

get_msg = Vector3()
receive_msg = Vector3()

# def msg_cb(msg):
#     receive_msg.data = msg


def clint():
    rospy.init_node('target_msg_clint', anonymous=True)
    get_msg_pub = rospy.Publisher('plane_position_yolo',Vector3,queue_size=1)
    # get_msg_sub = rospy.Subscriber('get_msg',Float32,msg_cb)
    rate = rospy.Rate(30)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接服务端
    s.connect(('127.0.0.1', 8000))
    count = 0
    while not rospy.is_shutdown():

        # 请求 | 发送数据到服务端
        print('######################')
        print('send hello')
        print('######################')
        s.send(b'hello')
        # 响应 | 接受服务端返回到数据
        receive_msg = s.recv(100).decode()

        target_msg = receive_msg.split(',')
        # 传给ros消息
        print('######################')
        print(target_msg)
        print('######################')
        nowtime = time()
        #print('time_now is {}, but pic_time is {},and end_time is {}'.format(nowtime,target_msg[3],target_msg[4]))


        #assume predict is pre_x pre_y pre_z,pre
        
        
        # initialize last_pos
        store = dict({'name':250})
        if len(store)<500:
            store[round(float(target_msg[3]),3)] = [float(target_msg[0])/1000,float(target_msg[1])/1000,float(target_msg[2])/1000]
        else:
            store=dict()
        get_msg.x = float(target_msg[0])/1000
        get_msg.y = float(target_msg[1])/1000
        get_msg.z = float(target_msg[2])/1000

        print('######################')
        rospy.loginfo(get_msg)
        print('######################')
        if get_msg.x < -1.2 and get_msg.x > -4 and get_msg.y > -1.0 and get_msg.y < 1.0 and get_msg.z > -1.0 and get_msg.z < 1:
            get_msg_pub.publish(get_msg)
        else:
            get_msg.x = 10 # warning the plane to hover
            get_msg.y = 0
            get_msg.z = 0
            get_msg_pub.publish(get_msg)
        rate.sleep()


# 关闭 socket
# s.close()
if __name__ == "__main__":
    clint()

