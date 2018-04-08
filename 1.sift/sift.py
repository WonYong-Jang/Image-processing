# -*-coding:utf-8-*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./image1.jpg',0)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hA, wA = img.shape[:2]
img = cv2.resize(img, (20, int(20*hA/wA)), cv2.INTER_AREA)
sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
kp, des1 = sift.detectAndCompute(img,None)
#result = img.copy()
cv2.drawKeypoints(gray,kp,img) #gray iamge keypoint들을 result에다가 output

#result2 = img.copy()
#cv2.drawKeypoints(gray,kp,result2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#keypoint 크기를 보여주겠다

fig = plt.figure("SIFT Feature", figsize=(21,7))
a = fig.add_subplot(1,3,1) #
a.set_title('Image')
plt.axis("off")
img1 = plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #opencv는 bgr이니까 원색을 보여주기 위해서

#b = fig.add_subplot(1,3,2)
#b.set_title('SIFT Default')
#plt.axis("off")
#img2 = plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))

#c = fig.add_subplot(1,3,3)
#c.set_title('RICH_KEYPOINTS')
#plt.axis("off")
#img3 = plt.imshow(cv2.cvtColor(result2,cv2.COLOR_BGR2RGB))

plt.show()


