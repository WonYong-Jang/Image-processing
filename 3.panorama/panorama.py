# -*-coding:utf-8-*-
import numpy as np
import cv2

class Stitcher: # self 자기를 표현,
    def orderPriority(self,OrdLeft, OrdRight, keyPoints, ratio=0.60, reprojThresh=4.0):
        # this function is to set whether stitching to left side of image or vice versa about two images
        # by decreasing ratio, it is to set more correct order about two images
        kps, features, gray = keyPoints[OrdLeft]
        kps_right, features_right, gray_right = keyPoints[OrdRight]

        M = self.matchKeypoints(kps, kps_right, # confirm keypoints
                                features, features_right, ratio, reprojThresh)
        re_M = self.matchKeypoints(kps_right, kps, # confirm the opposite case
                                features_right, features, ratio, reprojThresh)
        if M is None:
            matches,H,status =0,0,0
        if re_M is None:
            re_matches,H,status =0,0,0

        (matches, H, status) = M
        (re_matches, re_H, re_status) = re_M # the opposite case
        orderList = [] # array to set ordr of two images
        print(len(status), len(re_status)) # check length matched and set order two images
        if len(matches) > len(re_matches):
            orderList.append(OrdLeft)
            orderList.append(OrdRight)
            return orderList
        elif len(re_matches) > len(matches):
            orderList.append(OrdRight)
            orderList.append(OrdLeft)
            return orderList
        else: [] # if it can't order, return

    def detectKeyPoint(self, images, ratio=0.75, reprojThresh=4.0):

        descriptor = cv2.xfeatures2d.SIFT_create() # creat sift object
        keyPoints =[0 for i in range(4)]
        draw_keys = [0 for i in range(4)] # initialization
        for i in range(0, 4):
            gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            kps, features = descriptor.detectAndCompute(images[i], None)
            draw_keys[i] = kps # store in array to draw keypoints each image
            kps = np.float32([kp.pt for kp in kps])
            keyPoints[i] = (kps, features, gray) # manage with array to reuse keypoints
        order =[0,1,2,3] # initial array order
        order2=[] # store only two images to set order of images
        order3 = []
        order4 =[]
        order_set =[] # result order of images
        for i in range(0,3):
            kps, features, gray = keyPoints[0] # first input image keypoints

            kps_right, features_right, gray_right = keyPoints[i+1] # next input image
            M = self.matchKeypoints(kps, kps_right,
                features, features_right, ratio, reprojThresh)
            if M is None:
                continue
            (matches, H, status) = M
            print(len(matches)) # check length matched
            if len(matches)>=50:
                order2=self.orderPriority(0,i+1, keyPoints) # check two images in more detail
        #order.pop(order2[0]) # pop matched two images in initial order array
        #order.pop(order2[1]-1)
        for j in range(0,4):
            if order[j] != order2[0] and order[j] !=order2[1]:
                order3.append(order[j])
        order4 = self.orderPriority(order3[0],order3[1],keyPoints) # to set order second group about two images
        print(order4)
        print(order2) # check print

        kps, features, gray = keyPoints[order2[1]] # to set order of two groups about two images each

        kps_right, features_right, gray_right = keyPoints[order4[0]]
        M = self.matchKeypoints(kps, kps_right,
                                features, features_right, ratio, reprojThresh)
        if M is None:
            matches,H,status =0,0,0
        else: (matches, H, status) = M
        #print(len(matches))
        if M is not None and len(matches) >= 50:
            order_set.append(order2[0])
            order_set.append(order2[1])
            order_set.append(order4[0])
            order_set.append(order4[1])
        else: # 2 3 0 1
            order_set.append(order4[0])
            order_set.append(order4[1])
            order_set.append(order2[0])
            order_set.append(order2[1])
        print(order_set) # final order check

        imgCopy = [0 for i in range(4)] # draw keypoints of each image
        for i in range(0,4):
            imgCopy[i] = images[i].copy()
            kps,features, gray = keyPoints[i]
            cv2.drawKeypoints(gray, draw_keys[i], imgCopy[i])

        cv2.imshow('1',imgCopy[0])
        cv2.imshow('2', imgCopy[1])
        cv2.imshow('3', imgCopy[2])
        cv2.imshow('4', imgCopy[3])

        return order_set # return order of images

    def stitch(self, image_left, image_right, ratio=0.60, reprojThresh=4.0):
        # detect keypoints and extract local invariant descriptors from them

        (kps_right, features_right, gray) = self.detectAndDescribe(image_right)
        (kps_left, features_left,gray) = self.detectAndDescribe(image_left)

        # match features between the two images
        M = self.matchKeypoints(kps_left, kps_right,
        features_left, features_right, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched keypoints to create a panorama
        if M is None:
            return None
        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M
        #print(len(status))
        print(image_left.shape)
        print(image_right.shape)
        result = cv2.warpPerspective(image_right,H,
            (image_right.shape[1] + image_right.shape[1], image_right.shape[0])) # output 크기 정해준것

        result[0:image_left.shape[0],0:image_left.shape[1]] = image_left

        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image,None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps]) # 매치 키포인트를 할때 호모그래피할때 float해야 할수 있기 때문

        # return a tuple of keypoints and features
        return (kps, features, gray)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # get k=2 best matches
        rawMatches = matcher.knnMatch(featuresA,featuresB,2)
        matches = []

        # loop over the raw matches
        for m in rawMatches: # m[0] , m[1]
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio: # 두개를 비교
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4: # 호목래피는 최소 4개점, 두 겹치는 영상을 합칠때 잘 맞도록 변환시켜주는 호모그래피
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsB,ptsA,cv2.RANSAC,reprojThresh)
            #
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None
