#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:12:13 2020

@author: nagaraj
"""

import cv2
import numpy as np
from numpy import load
import collections
import matplotlib.pyplot as plt
import onnxruntime

cam = cv2.VideoCapture(0)
path = ''
img_path = ''
ort_session = onnxruntime.InferenceSession(path)
frame_size = (480,640,3)

lower_blue = (99,184,129)
upper_blue = (255,255,255)

pts = collections.deque(maxlen = 512)
print('Default frame size:',frame_size)
black_board = np.zeros(shape = frame_size, dtype=np.uint8)

digit = np.zeros((200, 200, 3), dtype=np.uint8)
pred_digit = ''

while cam.isOpened():
    
    ret, img = cam.read()
    if ret is False:
        continue
    #print('Capturing frame size:',img.shape)
    img= cv2.flip(img,1)
    scene = img.copy()
    blurred = cv2.GaussianBlur(scene,(11,11),0)
    #cv2.imshow('Gaussian',blurred)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    
    # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    # Masking
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    mask = cv2.erode(mask,kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    #mask = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=mask)
    
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center = None
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(scene, (int(x), int(y)), int(radius),(255, 0, 0), 2)
        cv2.circle(scene, center, 5, (0, 0, 255), -1)
        
        pts.appendleft(center)
        for i in range(1, len(pts)):
    		# if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = 7
            cv2.line(scene, pts[i - 1], pts[i], (0,0,255), 3)
            cv2.line(black_board, pts[i - 1], pts[i], (255, 255, 255), thickness)
            #cv2.circle(scene, pts[i], 1, (0, 0, 255), thickness)
    
    elif len(cnts) == 0:
        if len(pts) != 0:
            cv2.imshow('board',black_board)
            blackboard_gray = cv2.cvtColor(black_board, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            ret1, thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(blackboard_cnts) >= 1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)  # Each list elements are compared with respect to the contour area, the shape (contours) with max area is taken.
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 1600:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y-30:y + h + 30, x-30:x + w + 30]
                    cv2.imshow('Digit',digit)
                    newImage = cv2.resize(digit, (28, 28))
                    newImage = np.array(newImage, dtype = np.float32)
                    #newImage = np.stack((newImage,)*3,axis=-1)
                    #newImage = newImage.flatten()
                    
                    #cv2.imwrite(img_path+'image_'+str(np.random.randint(0,high=100,size=1)[0])+'.jpg',newImage)    # To save the digits you have written.
                    #print(newImage.shape)
                    newImage = newImage.reshape(1,1,28,28)
                    
                    ort_inputs = {ort_session.get_inputs()[0].name: newImage}     # The input must be a numpy array 
                    ort_outs = ort_session.run(None, ort_inputs)
                    predicted = ort_outs[0]
                    pred_digit = ''
                    a = []
                    a =list(np.exp(predicted).squeeze())
                    pred_digit = a.index(max(a))
                    print(predicted)
                    print(np.exp(predicted))
                    print('digit predicted:',str(pred_digit))
            pts = collections.deque(maxlen = 512)
            black_board = np.zeros(shape = frame_size, dtype=np.uint8)
    
    cv2.putText(scene, "MNIST Onnx : " + str(pred_digit), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.imshow('Masked',mask)
    cv2.imshow('Scene',scene)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()