# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:18:47 2019

@author: ZZ
"""

import cv2
import keras
import os
from keras import models
import time
import numpy as np
def image_process(image):
    image=cv2.imread(image)
    image=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
    image=image.astype("float")/255
    image=np.expand_dims(image,axis=0)
    return image
def result_process(result):
    num_path="d:/pics/mask_test/refine/"
    cow_nums=os.listdir(num_path)
    tmp=0
    for i in range(len(result[0])):
        if(result[0][i]>=result[0][tmp]):
            tmp=i
    return cow_nums[tmp] 
def predict(img_path,model_path):
    image=image_process(img_path)
    model=models.load_model(model_path)
    result=model.predict(image)
    num=result_process(result)
    return num
start=time.time()
img="d:/pics/mask_test/refine/463/0655.jpg_1.jpg"
model="d:/model/arimura80.h5"
print(predict(img,model))
end=time.time()
print(end-start)
