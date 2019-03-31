# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:50:24 2019

@author: ohwada
"""

import cv2
import time
def main():
    cap=cv2.VideoCapture("http://192.168.137.88:8000/stream.mjpg")
    cap.set(3,1920)
    cap.set(4,1080)
    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    count=0
    name=time.asctime(time.localtime(time.time()))
    name=name[4:7]+"-"+name[8:10]+"-"+name[11:13]+"-"+name[14:16]
    path="C:/video/"+name+".avi"
    fb=open("c:/video/record.txt","a")
    fb.writelines(name)
    fb.close()
    video=cv2.VideoWriter(path,fourcc,10,(1920,1080))
    while True:
        ret,fra=cap.read()
        if(ret==True):
            cv2.imshow("img",fra)
            cv2.waitKey(1)
            video.write(fra)
            count+=1
        if(count>600):
            video.release()
            count=0
            name=time.asctime(time.localtime(time.time()))
            name=name[4:7]+"-"+name[8:10]+"-"+name[11:13]+"-"+name[14:16]
            path="C:/video/"+name+".avi"
            video=cv2.VideoWriter(path,fourcc,10,(1920,1080))
            fb=open("c:/video/record.txt","a")
            fb.writelines(name)
            fb.close()
if __name__=="__main__":
    main()
            