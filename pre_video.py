import cv2

import numpy
import os

capture1 = cv2.VideoCapture('1.avi')

capture2 = cv2.VideoCapture('2.avi')


dir = './after/'
i=0
j=467
success, frame = capture2.read()
print("success",success)
while success :
    i = i + 1
    # if i % ==0:
    name = dir + 'lanenet_1015_' + str(j) + '.png'
    cv2.imwrite(name,frame)
    j = j+1
    print("saving ",name)
   
    success, frame = capture2.read() #获取下一帧
