import cv2
import numpy as np
import os
import sys

global edge
global img

def imgSetter(img, edge) :
    edge = edge
    edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
    img = img
    return None

def removeNoise():
    edge = cv2.GaussianBlur(globals.edge, (3, 3), 0) # 블러링
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # 마스크를 지정한 것임
    edge = cv2.filter2D(edge, -1, kernel)# 이미지샤프닝
    #cv2.imwrite("./resultTestImage/shapeDetectionTest/removeNoise/" + str(i) + ".jpg", edgeImg)
    return None

def edgeExpand(idx) :
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    edge = cv2.morphologyEx(globals.edge, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite("./resultTestImage/shapeDetectionTest/edgeExpand/" + str(idx) + ".jpg", edge)
    return None

def shapeDraw(idx) :

    _, contours, _ = cv2.findContours(globals.edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    cv2.drawContours(img, [cnt], 0, (0,255,0),  3)
    cv2.imwrite("./resultTestImage/shapeDetectionTest/shapeDraw/" + str(idx) + ".jpg", img)

"""
    for i in range(0, sys.getsizeof(contours)) :
        #rect = cv2.minAreaRect(contours[i])
        #areaRatio = abs(cv2.contourArea(contours[i])) / sys.getsizeof(rect).
        cv2.drawContours(contoursImg, [contours[0]], 0, (0,255,0), 3)
    cv2.imwrite("./resultTestImage/shapeDetectionTest/shapeDraw/" + str(i) + ".jpg", img)
"""
#def squareDetect() :

def run(idx) :
    removeNoise()
    edgeExpand(idx)
    shapeDraw(idx)

"""
if __name__ == "__main__" :
    path = './resultTestImage/metro/edge'
    i = 0  # 결과영상 이름 찍는 용도로 쓰이는 변수임.
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':
                path += '/' + file
                print(path)
#                originalImg = edgeDetect.targetTemp
                imgSetter(edgeDetect.targetGetter(), edgeDetect.edgeGetter())
                edgeImg = removeNoise(edgeImg)
                edgeImg = edgeExpand(edgeImg)
                shapeDraw(edgeImg,i)#, originalImg)
                path = './resultTestImage/metro/edge'
                i += 1
"""