# coding=utf-8
import cv2
import os
import threading
from threading import Lock, Thread

video_path = "C:/Users/Salieri/Desktop/data/videos/"
pic_path = "E:/data/pictures/"
filelist = os.listdir(video_path)  # 返回指定的文件夹下包含的文件或文件夹名字的列表，这个列表按字母顺序排序。


def video2pic(filename):
    # print(filename)
    cnt = 0
    dnt = 0
    if os.path.exists(pic_path + str(filename)):
        pass
    else:
        os.mkdir(pic_path + str(filename))
    cap = cv2.VideoCapture(video_path + str(filename))  # 读入视频
    while True:
        # 按帧读取视频
        ret, image = cap.read()  # ret: 读帧正确返回True，读到文件结尾返回False; image: 读到的帧图像
        if image is None:
            break
        # show a frame

        w = image.shape[1]  # 宽
        h = image.shape[0]  # 高
        if (cnt % 3) == 0:
            cv2.imencode('.jpg', image)[1].tofile(pic_path + str(filename) + '/' +str(filename[:-4])+'_'+ str(dnt) + '.jpg')
            # cv2.imwrite('C:/Users/JiangHuiQi/Desktop/pic/' + str(filename) + '/' + str(dnt) + '.jpg', image) #含中文路径，不可行
            print(pic_path + str(filename) + '/' + str(dnt) + '.jpg')
            dnt = dnt + 1
            # 显示
            # cv2.namedWindow('sliding_slice',0)
            # cv2.imshow('image', image)
            # cv2.waitKey(1000)
        cnt = cnt + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):   #　等待键盘输入, 参数是1，表示延时1ms切换到下一帧图像
            break
    cap.release()  # 释放摄像头，关闭所有图像窗口。


if __name__ == '__main__':
    for filename in filelist:
        threading.Thread(target=video2pic, args=(filename,)).start()