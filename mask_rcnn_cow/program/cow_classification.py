# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image 
import cv2
import keras 
from keras import models
import numpy as np
import os
import time
def image_process(img):
    img=cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
    img=np.expand_dims(img,axis=0)
    return img
def result_process(result):
    index_path="/home/pjzz/index"
    index=os.listdir(index_path)
    tmp=0
    for i in range(len(index)):
        if(result[0][i]>=result[0][tmp]):
            tmp=i
    return index[tmp]
cap=cv2.VideoCapture(0)
model=models.load_model("/home/pjzz/model/arimura80.h5")
font=cv2.FONT_HERSHEY_SIMPLEX
color=[255,255,255]
count=0
number="wait....."
while True:
    cv2.namedWindow("img",0)
    cv2.resizeWindow("img",640,480)
    if(count>20):
        end2=time.time()
        print(end2-start)
        count=0
    if(count==0):
        ret,fra=cap.read()
        if(ret==True):
            start=time.time()
            image=image_process(fra)
            result=model.predict(image)
            number=result_process(result)
            res_img=cv2.putText(fra,number,(320,240),font,3,color,3,cv2.LINE_AA)
            end1=time.time()
            print(end1-start)
            cv2.imshow("img",res_img)
            cv2.waitKey(1)
    if(0<count<20):
        ret,fra==cap.read()
        if(ret==True):
            res_img=cv2.putText(fra,number,(320,240),font,3,color,3,cv2.LINE_AA)
            cv2.imshow("img",res_img)
            cv2.waitKey(1)
    count+=1       
cv2.destroyAllWindows()