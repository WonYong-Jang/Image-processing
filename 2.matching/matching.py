import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # testImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None) # 키포인터와 디스크립터까지 찾아주는 것
kp2, des2 = sift.detectAndCompute(img2,None) #디스크립터 : 해당 키포인터 위치에서 추출한 지역적 영상 특징 정보

# BFMatcher with default params
bf = cv2.DescriptorMatcher_create("BruteForce") # 전부 다 비교하는 알고리즘
matches = bf.knnMatch(des1,des2,k=2) # 매치 방법중 1(찾을 이미지),2

# Apply ratio test
good = [] #두개 찾은것중 한개씩 불러와서 거리 비교 0.75보다 작은 경우 신뢰할수 있는 디스크립터로 해서 추가
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

#output image initialization
h1, w1 = img1.shape[:2] # 크기가 두 사진이 다르니까 맞춰줘서 한번에 출력하기 위해
h2, w2 = img2.shape[:2]
height = max(h1, h2)
width = w1 + w2
channels = 3

img3 = np.zeros((height, width, channels),dtype=np.uint8)

# cv2.drawMatchesKnn expects list of lists as matches.
cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2) #플래그 2번 피피티 보면 한번 나온건 안그리겠다

# visualization
plt.figure("Feature matching")
plt.title('Feature matching')
plt.axis("off")
fig = plt.imshow(img3)
plt.show()
