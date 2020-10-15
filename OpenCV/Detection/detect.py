
import numpy as np
import cv2
import tkinter

font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体样式

def ParameterChange():#多线程才能实现了

    myWindow = tkinter.Tk()
    #设置标题
    myWindow.title('configMenu')
    #设置窗口大小
    width = 380
    height = 300

    #获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
    screenwidth = myWindow.winfo_screenwidth()
    screenheight = myWindow.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
    myWindow.geometry(alignstr)

    #设置窗口是否可变长、宽，True：可变，False：不可变
    myWindow.resizable(width=False, height=True)
    Minvalue = tkinter.Scale(myWindow,from_=1,to=1000,resolution=10,orient=tkinter.HORIZONTAL,label='minRadius') #Scale组件
    Minvalue.pack()
    Maxvalue = tkinter.Scale(myWindow,from_=1,to=1000,resolution=10,orient=tkinter.HORIZONTAL,label='maxRadius') #Scale组件
    Maxvalue.pack()
    #进入消息循环
    myWindow.mainloop()

    min_r,max_r = Minvalue.get(),Maxvalue.get()

def find_circles(edges,min_r,max_r,frame):
    r = 0
    x = 0
    y = 0
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=100, param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is not None:  # 如果识别出圆
        print(circles)
        index = 0
        for i in range(0,len(circles[0])):
            #  获取圆的坐标与半径
            if(circles[0][i][2]<min_r or circles[0][i][2]>max_r):
                continue
            elif(circles[0][i][2]>r):
                index = i;
                r = circles[0][i][2];
        r = int(r)
        x = int(circles[0][index][0])
        y = int(circles[0][index][1])
        cv2.circle(frame, (x, y), r, (0, 0, 255), 3)  # 标记圆
        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)  # 标记圆心
        text = 'x:  '+str(x)+' y:  '+str(y)
        cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA, 0)  # 显示圆心位置
        return x,y,r
    else:
        # 如果识别不出，显示圆心不存在
        cv2.putText(frame, 'x: None y: None', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA, 0)
    return x,y,r

def shot():
    cap = cv2.VideoCapture(0)
    ret = cap.set(3, 640)  # 设置帧宽
    ret = cap.set(4, 480)  # 设置帧高
    kernel = np.ones((5, 5), np.uint8)  # 卷积核
    low_range1 = np.array([0, 43, 46])
    high_range1 = np.array([10, 255, 255])
    low_range2 = np.array([156, 43,46])
    high_range2 = np.array([180, 255, 255])

    low_range = np.array([35, 50, 100])
    high_range = np.array([77, 255, 255])

    ret, frame = cap.read()

    while ret:  # 循环读取视频帧
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间
        #mask 红色
        th1 = cv2.inRange(hsv, low_range1, high_range1)
        th2 = cv2.inRange(hsv, low_range2, high_range2)
        th = cv2.inRange(hsv, low_range, high_range)
        mask = th
        #mask = cv2.add(th1,th2)
         # 形态学开运算
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


        bulred = cv2.GaussianBlur(opening,(3,3),0)
        egs = cv2.Canny(bulred,50,100)
        find_circles(egs,50,180,frame)

        cv2.imshow('live camera', frame)
        cv2.imshow('edges', egs)
        cv2.imshow('opening', opening)
        ret, frame = cap.read()
    cap.release()

if __name__ == "__main__":
    shot()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
