import cv2;
import numpy as np
from matplotlib import pyplot as plt
import copy

img = cv2.imread('images/Image1.pgm', cv2.IMREAD_GRAYSCALE);
if img is not None: print('Image Loading Completed'); # 불러온 이미지가 정상적으로 읽혔는지 체크
else: print('Failed to Image Loading'); # 에러가 있을경우는 None 을 출력함

height, width = img.shape; #순서대로 정보가 들어감, shape 색깔일때는 채널이 3개 까지 포함

label =0; # 라벨링 해주기 위해 초기화

print(height,width); # 출력으로 값 확인
print(img[0][0]); # 색이라면 B G R 순서대로

def labeling(label,img): # 4연결성에 의한 라벨링
    for i in range(0,height):
        for j in range(0,width):
            if img[i][j] == 255:
                if i-1 >= 0 and img[i-1][j] != 0 and img[i-1][j] != 255:
                    img[i][j] = img[i-1][j]; # 만난 값의 위쪽 값이 0도 아니고 255도 아니면 라벨링되어있단 뜻이니깐 연결
                elif i+1 < height and img[i + 1][j] != 0 and img[i+1][j] != 255:
                    img[i][j] = img[i+1][j]; # 똑같은 원리로 아래쪽 확인
                elif j-1 >= 0 and img[i][j-1] != 0 and img[i][j-1] != 255:
                    img[i][j] = img[i][j-1]; # 똑같은 원리로 왼쪽 확인
                elif j+1 < width and img[i][j+1] != 0 and img[i][j+1] != 255:
                    img[i][j] = img[i][j+1]; # 똑같은 원리로 오른쪽 확인
                else:
                    label += 1; # 4 연결성 확인했는데 처음 보는 값이면 라벨링 하나 추가해서 저장
                    img[i][j] = label;

def dilationFunc(img):
    tempList = copy.deepcopy(img); # 받아온 이미지배열을 복사하고 tempList 를 이용해서 check 하면서 img 변경
    myFilter =[ [0,255,0], # 4연결성 필터 생성
                [255,255,255],
                [0,255,0]]
    for i in range(0,height):
        for j in range(0,width):
            if tempList[i][j] == 255: # 255만나게 되면 조건에 맞는지 확인하고 필터를 적용할 준비
                for k in range(-1,2):
                    for l in range(-1,2):
                        if i+k<0 or i+k>=height or j+l<0 or j+l >=width \
                                or img[i+k][j+l] == 255: continue; # 범위에 벗어나거나 이미 255 이면 continue
                        if myFilter[k+1][l+1] != 255: continue; # 4연결성이므로
                        img[i+k][j+l] = myFilter[k+1][l+1]; # 조건 만족하면 필터 적용
            """   조건문으로만 4연결성 필터를 만든 소스
                if i-1 >= 0:
                    img[i-1][j] = 255;
                if i+1 < height:
                    img[i+1][j] = 255;
                if j-1 >= 0:
                    img[i][j-1] = 255;
                if j+1 < width:
                    img[i][j+1] = 255;
            """
def erosionFunc(img):
    tempList = copy.deepcopy(img);
    myFilter = [[0,255,0],  # 4연결성 필터 생성
                [255,255,255],
                [0,255,0]]
    count=0; # 4연결성 확인을 위한 count
    for i in range(0,height):
        for j in range(0,width):
            if tempList[i][j] == 255:
                for k in range(-1,2):
                    for l in range(-1,2):
                        if i+k<0 or i+k>=height or j+l<0 or j+l >=width \
                                or (i == i+k and j == j+l): continue; #범위에 벗어나거나 자기 자신일때
                        if myFilter[k+1][l+1] == 255 and myFilter[k+1][l+1] == tempList[i+k][j+l]:
                            count+=1; # 조건에 만족하는 경우 count
            if count != 4: img[i][j] = 0; # 4연결성 만족하지 않는 경우 0으로
            count =0; #count 초기화

            """ 조건문으로만 4연결성 필터를 만든 소스
                if i-1 >= 0 and tempList[i-1][j]== 255:
                    count+=1;
                if i+1 < height and tempList[i+1][j] == 255:
                    count+=1;
                if j-1 >= 0 and tempList[i][j-1] == 255:
                    count+=1;
                if j+1 <width and tempList[i][j+1] == 255:
                    count+=1;
                if count != 4: img[i][j] = 0;
                count = 0;
            """

#labeling(label,img);

#dilationFunc(img);

erosionFunc(img);


""" 기존에 정의되어 있는 라이브러리의 결과와 직접 짠 코드의 결과를 비교하기 위해
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#dilation = cv2.dilate(img,kernel,iterations=1);
#erosion = cv2.erode(img, kernel, iterations=1)
#cv2.imshow('dilation',dilation);
#cv2.imshow('erosion',erosion);
"""

cv2.imshow('input image',img);
cv2.waitKey(0);
cv2.destroyAllWindows();