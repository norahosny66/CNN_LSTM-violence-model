#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from mamonfight22 import *

model = mamon_videoFightModel2(tf,wight='/content/drive/MyDrive/mamonbest947oscombo.hdf5')

cap = cv2.VideoCapture('hospital.mp4')
i = 0
frames = np.zeros((30, 160, 160, 3), dtype=np.float)
old = []
j = 0
while(True):
    ret, frame = cap.read()
  
    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX
    if i > 29:
        ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
        ysdatav2[0][:][:] = frames
        predaction = pred_fight(model,ysdatav2,acuracy=0.96)
        if predaction[0] == True:
            cv2.putText(frame, 
                'Violance Deacted  ... Violence .. violence', 
                (50, 50), 
                font, 3, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
      
            print('Violance detacted here ...')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vio = cv2.VideoWriter("./videos/output-"+str(j)+".avi", fourcc, 10.0, (fwidth,fheight))
            #vio = cv2.VideoWriter("./videos/output-"+str(j)+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (300, 400))
            for frameinss in old:
                vio.write(frameinss)
            vio.release()
        i = 0
        j += 1
        frames = np.zeros((30, 160, 160, 3), dtype=np.float)
        old = []
    else:
        frm = resize(frame,(160,160,3))
        old.append(frame)
        fshape = frame.shape
        fheight = fshape[0]
        fwidth = fshape[1]
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        
        i+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

import time
millis = int(round(time.time() * 1000))
print("started at " , millis)
vid = video_mamonreader(cv2,'dvpalfight.dvd')
millis2 = int(round(time.time() * 1000))
print("time processing " , millis2 - millis)

datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)

datav[0][:][:] = vid

millis = int(round(time.time() * 1000))
print("started at " , millis)
print(pred_fight(model,datav,acuracy=0.6))
millis2 = int(round(time.time() * 1000))
print("time processing " , millis2 - millis)

def video_mamonreader(cv2,filename):
    frames = np.zeros((30, 160, 160, 3), dtype=np.float)
    i=0
    print(frames.shape)
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(160,160,3))
    frm = np.expand_dims(frm,axis=0)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    print("reading video")
    while i < 30:
        rval, frame = vc.read()
        frm = resize(frame,(160,160,3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
    return frames

novid = video_mamonreader(cv2,'nofight.mp4')

nodatav = np.zeros((1, 40, 170, 170, 3), dtype=np.float)

nodatav[0][:][:] = novid

pred_fight(model,nodatav,acuracy=0.6)

ysvid2 = video_mamonreader(cv2,'hdfight.mp4')

ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float)

ysdatav2[0][:][:] = ysvid2

import time

millis = int(round(time.time() * 1000))
print("started at " , millis)
predaction = pred_fight(model,ysdatav2,acuracy=0.9)
print(predaction)
millis2 = int(round(time.time() * 1000))
print("time processing " , millis2 - millis)

if predaction[0] == True:
    print('violence')

novid3 = video_mamonreader(cv2,'golsss.mp4')

nodatav3 = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
nodatav3[0][:][:] = novid3

millis = int(round(time.time() * 1000))
print("started at " , millis)
print(pred_fight(model,nodatav3,acuracy=0.8))
millis2 = int(round(time.time() * 1000))
print("time processing " , millis2 - millis)

novid4 = video_mamonreader(cv2,'dvpalfight.dvd')

nodatav4 = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
nodatav4[0][:][:] = novid4

millis = int(round(time.time() * 1000))
print("started at " , millis)
print(pred_fight(model,nodatav4,acuracy=0.9))
millis2 = int(round(time.time() * 1000))
print("time processing " , millis2 - millis)
