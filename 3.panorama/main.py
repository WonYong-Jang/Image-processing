# import package
# -*-coding:utf-8-*-

from panorama import Stitcher
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser() #객체 선언
ap.add_argument("-fi", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
ap.add_argument("-t", "--third", required=True,
	help="path to the third image")
ap.add_argument("-fo", "--fourth", required=True, # to input four pictures
	help="path to the fourth image")
args = vars(ap.parse_args())

# load the four images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageC = cv2.imread(args["third"])
imageD = cv2.imread(args["fourth"])
hA, wA = imageA.shape[:2] # height and width of each image
hB, wB = imageB.shape[:2]
hC, wC = imageC.shape[:2]
hD, wD = imageD.shape[:2]

imageA = cv2.resize(imageA, (400, int(400*hA/wA)), cv2.INTER_CUBIC) # resize
imageB = cv2.resize(imageB, (400, int(400*hB/wB)), cv2.INTER_CUBIC)
imageC = cv2.resize(imageC, (400, int(400*hC/wC)), cv2.INTER_CUBIC)
imageD = cv2.resize(imageD, (400, int(400*hD/wD)), cv2.INTER_CUBIC)
images = [imageB,imageA, imageD, imageC] # order of original each image => CDAB
order_set =[] # to correct order of each image
# stitch the images together to create a panorama
stitcher = Stitcher() # declartion

order_set = stitcher.detectKeyPoint(images) # return correct order of images

result = stitcher.stitch(images[order_set[0]],images[order_set[1]]) # group by two images and switch them
result2 = stitcher.stitch(images[order_set[2]],images[order_set[3]])


result3 = stitcher.stitch(result,result2)

cv2.imshow("Result", result) # first stitching
cv2.imshow("Result2",result2) # second stitching

cv2.imshow("Result3",result3) # stitching the first and second one

cv2.waitKey(0)
