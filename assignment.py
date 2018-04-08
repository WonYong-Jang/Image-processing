# -*-coding:utf-8-*-
import cv2;
import numpy as np

img = cv2.imread('images/cameraman.png',cv2.IMREAD_GRAYSCALE); # 해당 디렉토리에 있는 사진을 GRAYSCALE로 읽어옴
if img is not None: print('Image Loading Completed'); # 불러온 이미지가 정상적으로 읽혔는지 체크
else: print('Failed to Image Loading'); # 에러가 있을경우는 None 을 출력함

maxSize = 256;  # 히스토 그램 0 ~255
sum =0;
sumB =0;
wBackground =0;
wForeground =0;
varMax =0;
threshold =0; # 임계값
total = img.size; # 256 * 256  전체 픽셀 수  정규화를 위하여
#print(total)
hist = [0]*maxSize; # 0으로 초기화

print(img.shape);
height, width= img.shape # 256 * 256

for i in range(0,height):
    for j in range(0,width):
       hist[img[i][j]] +=1;  # 전체 영상을 확인하면서 히스토그램을 계산한다.

plt.hist(img.ravel(),256,[0,256]); # 계산한 히스토 그램 확인
plt.show();

for i in range(0,maxSize):
    sum += (i * hist[i]); # 전체 합을 미리 구해 놓는다


for i in range(0,maxSize): # 효율적인 오추알고리즘 방법을 선택했습니다.

    wBackground += hist[i];            # Weight Background
    if wBackground == 0: continue; # 히스토그램 처음부터 더해나가는데 0이면 계산할 필요 없으니 continue

    wForeground = total - wBackground;          # Weight Foreground
    if wForeground==0: break; # background == total 이란 소리니까 for문 빠져나감

    sumB +=((i) * hist[i]); # 평균을 구하기 위해서

    mB = sumB / wBackground;           # Mean Background
    mF = (sum - sumB) / wForeground;   # Mean Foreground

    varBetween = (wBackground/total) * (1- wBackground/total) * (mB -mF)**2; #교수님 ppt 공식

    if varBetween > varMax:   # 가장 큰 값을 찾는다
        varMax = varBetween;
        threshold = i; # 가장 큰 값의 배열 인덱스 기억해 둔다.

print(threshold); # 나온 임계값 확인
# threshold(이미지 , 임계값, 임계값 보다 클경우 적용되는 최대값, 임계값 적용 방법 또는 스타일)
ret, thr = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY); # 검정색 흰색이 명확하게 구분 binary

ret1, thr1 = cv2.threshold(img,0,255, cv2.THRESH_OTSU); # 기존 오추 라이브러리로 임계값 확인 후 제가 구한 임계값이 맞는지 확인 하였습니다.
print(ret1);
cv2.imshow('cameraman',thr);
cv2.waitKey(0);
cv2.destroyAllWindows();


















