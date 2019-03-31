# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:16:53 2019

@author: ohwada
"""

import cv2
import time
import zipfile
import os
cap=cv2.VideoCapture("http://192.168.137.88:8000/stream.mjpg")
count=0
name=time.asctime(time.localtime(time.time()))
name=name[4:7]+"-"+name[8:10]+"-"+name[11:13]+"-"+name[14:16]
f = zipfile.ZipFile('c:/video/buffer/{}.zip'.format(name),'w',zipfile.ZIP_DEFLATED)
while True:
    if(count<1000):
        ret,fra=cap.read()
        if(ret==True):
            try:
                cv2.imwrite("c:/video/test/sample{}.jpg".format(str(count)),fra)
                f.write("c:/video/test/sample{}.jpg".format(str(count)))
                os.remove("c:/video/test/sample{}.jpg".format(str(count)))
                count+=1
            except Exception:
                continue
    else:
        count=0
        f.close()
        print("generate success")
        name=time.asctime(time.localtime(time.time()))
        name=name[4:7]+"-"+name[8:10]+"-"+name[11:13]+"-"+name[14:16]
        f = zipfile.ZipFile('c:/video/buffer/{}.zip'.format(name),'w',zipfile.ZIP_DEFLATED)
        continue