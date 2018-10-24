
import cv2
import os
import os.path
import numpy as np
import re
rootdir = "./testdata/videotest/"
aa="./testdata/bi/"
bb="./testdata/in/"
#
s=[]
file_object = open('./testdata/train.txt','w')
# # file_object.write("aaaaaa"+ '\n')
i=0
c=[]
for parent,dirnames,filenames in os.walk(rootdir):
#     for i in range(len(filenames)):
#         aa=re.findall("\d+",filenames[i])
#         print (aa[0])
#         os.rename(rootdir+filenames[i],rootdir+str(aa[0])+".png")
#
    for filename in filenames:
#
        fullname= rootdir+filename
# #         # img = cv2.imread(fullname)
        newa=aa+"0000.png"
        newb=bb+"0000.png"

#
# #         file_object.write(fullname + '\n')
        newname = fullname+' '+newa+' '+newb+' '
        print(newname)
        file_object.write(newname+ '\n')
file_object.close()

# file_object=open('./testdata/test_video_imgs/train.txt')
# lines = file_object.readlines()
# lines = file_object.readlines()
# for line in lines:
#     rs = line.rstrip('\n') #去除原来每行后面的换行符，但有可能是\r或\r\n
#     newname=rs.replace(rs,fullname+rs
#     	+' '+fullname+rs+' '+
#     	fullname+rs)
#     print(newname)
# file_object.close()
#
#
#

# import cv2
#
# import os
#
# file_object = open('./testdata/train.txt','r')
# lines = file_object.readlines()
# # print(lines)
#
# for line in lines:
#     img = cv2.imread(line,cv2.COLOR_BGR2GRAY)
#     new=