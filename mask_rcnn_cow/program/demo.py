pu\sys# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 23:14:23 2018

@author: ZZ
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pandas as pd
# Root directory of the project
ROOT_DIR = os.path.abspath("D:/python/practice/Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from pycocotools.coco import COCO

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
from coco import CocoConfig
class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
cow_df=pd.read_excel("d:/pics/gopro.xlsx",sheet_name="right")
cow_nums=list(cow_df["cow_id"][16:])
for i in range(len(cow_nums)):
    cow_nums[i]=str(cow_nums[i])
dst_dir="d:/pics/mask_test/refine/"
'''for cow_num in cow_nums:
    cow_num=str(cow_num)
    dst_file=os.path.join(dst_dir,cow_num)
    os.mkdir(dst_file)
'''
for cow_num in cow_nums:
    cow_num=str(cow_num)
    base_dir="d:/pics/mask_test/"
    image_dir=base_dir+cow_num+"/"
    imgnames=os.listdir(image_dir)
    dst_path=dst_dir+cow_num+"/"
    for imgname in imgnames:
        image_path=image_dir+imgname
        image=cv2.imread(image_path)
    # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,r['scores'])
        N = r['rois'].shape[0]
        colors = visualize.random_colors(N)
        rois_cow=[]
        for i in range(N):
            if(class_names[r['class_ids'][i]]=="cow" 
              and (r["rois"][i][2]-r["rois"][i][0])>700 
              and (r["rois"][i][3]-r["rois"][i][1])>400
              and (r["rois"][i][3]-r["rois"][i][1])<900):
                rois_cow.append(i)
        if(len(rois_cow)>3):
            break
        for i in rois_cow:
            dst_path=dst_dir+cow_num+"/"          
            if((r["rois"][i][3]+r["rois"][i][1])/2<480):
                dst_path=dst_dir+cow_nums[cow_nums.index(cow_num)-1]+"/"
            if((r["rois"][i][3]+r["rois"][i][1])/2>1440):
                dst_path=dst_dir+cow_nums[cow_nums.index(cow_num)+1]+"/"
            #result_image = image.copy()
            color = colors[i]
            rgb = (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255))
    
    #Rect
    #result_image = visualize.draw_box(result_image, r['rois'][i], rgb)
    
    #Label & Score
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = class_names[r['class_ids'][i]] + ':' + str(r['scores'][i])
    #result_image = cv2.putText(result_image, text,(r['rois'][i][1],r['rois'][i][0]), font, 0.8, rgb, 2, cv2.LINE_AA)
            print(class_names[r['class_ids'][i]])
    #Mask
            mask = r['masks'][:, :, i]
            class_name=class_names[r['class_ids'][i]]
            #result_image = visualize.apply_mask(result_image, mask, color)
            #cv2.imwrite(dst_path+"mrcnn_1_"+str(i)+".jpg",result_image)
            refine_img=image.copy()
            roi=r["rois"][i]
            refine=np.zeros((roi[2]-roi[0],roi[3]-roi[1],3))
            for m in range(roi[2]-roi[0]):
                for j in range(roi[3]-roi[1]):
                    for k in range(3):
                        refine[m][j][k]=refine_img[roi[0]+m][roi[1]+j][k]
            cv2.imwrite(dst_path+imgname+"_"+str(i)+".jpg",refine)
            '''
            for c in range(3):
                if(c==1):
                    refine_img[:, :, c] = np.where(mask == 1,refine_img[:, :, c],255)
                else:
                    refine_img[:, :, c] = np.where(mask == 1,refine_img[:, :, c],0)
        '''
       
