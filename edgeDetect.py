import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import datetime

targetTemp = []
edgeTemp = []

def extract_color(img, hsv):
    # define range of yellow lor in HSV
    lower_yellow = np.array([10,0,15])
    upper_yellow = np.array([50,255,180])    # VSH 순임
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_not(mask, mask)
    res = cv2.bitwise_and(img, img, mask = mask)

#    cv2.imwrite("./test/ex1.jpg", res)
    return res


def removeNoise(target):
    target = cv2.GaussianBlur(target, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # 마스크를 지정한 것임
    target = cv2.filter2D(target, -1, kernel)# 이미지샤프닝
#    targetView = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#    plt.subplot(3, 3, 3), plt.imshow(targetView), plt.title('blur'), plt.axis("off")
    return target


def back_project(hsv_r, hsv_t):
    roihist = cv2.calcHist([hsv_r], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 샘플영상 히스토그램 만들어줌

    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX) # 정규화 수행
    dst = cv2.calcBackProject([hsv_t], [0, 1], roihist, [0, 180, 0, 256], 1)
#    plt.subplot(3, 3, 4), plt.imshow(dst, 'gray'), plt.title('backprojection'), plt.axis("off")
#    cv2.imwrite("./test/backprojection.jpg", dst)

    return dst

def convolution(dst):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # backprojection만 수행한 결과영상을 보면 불명확하게 점자블록 범위가 지정되어 나오는데 그걸 컨볼루션 연산으로
    # 확장시키면 확실하게 점자블록 부분을 구분할 수 있게됨
    # 그걸 위한 부분이 이 함수임
    cv2.filter2D(dst, -1, disc, dst)
#    cv2.imwrite("./test/convolution.jpg", disc)

#    plt.subplot(3, 3, 5), plt.imshow(dst, 'gray'), plt.title('convolution'), plt.axis("off")

    return disc


def thresholding(disc, dst):
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, disc, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, disc, iterations=2)
    # 컨볼루션 연산된 영상을 50~255까지의 밝기값을 가진 픽셀을 255로 바꿈(이진영상을 만듬)
    # 이 이진영상은 영상분할을 위한 마스크영상임
#    cv2.imwrite("./test/thresh.jpg", thresh)

#    plt.subplot(3, 3, 6), plt.imshow(thresh), plt.title('thresholding'), plt.axis("off")
    return thresh


def merge(thresh, target):
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(target, thresh)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # and 연산으로 이진영상(마스크)와 원본영상 합성
#    plt.subplot(3, 3, 7), plt.imshow(res), plt.title('segmentationResult'), plt.axis("off")

    return res


def edgeDetect(res):
    edge = cv2.Canny(res, 100, 200)
    #경계선 검출
#    plt.subplot(3, 3, 8), plt.imshow(edge), plt.title('edgeDetect'), plt.axis("off")
#    cv2.imwrite("./test/edge.jpg", edge)
    return edge

def trim(target) :
#    img = cv2.imread(path, cv2.IMREAD_COLOR)
    x = 180;  y = 240;
    w = 120;   h = 160; # x, y좌표를 기준으로 가로로 120px, 세로로 160px을 범위로 지정
    img_trim = target[y: y+h , x: x+w] # 지정된 범위의 영상을 잘라옴
    return img_trim
    # 이게 내가 말하는 샘플영상인데 우리가 640x480 영상을 사용하기 때문에 중앙에 있는 범위를 가져옴
    # 영상 가운데에 점자블록이 위치하고 있다면 사용자가 점자블록을 제대로 짚고 있는 것으로 가정한것
    # 물론 이 범위는 우리가 프로토타입을 만들면서 변경될 수 있는 부분임
    # 내가 가지고 있는 70여장의 샘플영상을 돌려본 결과 1장의 영상도 빠짐없이 점자블록을 캐치하였음

def targetSetter(target) :
    targetTemp.append(target)

def edgeSetter(edge) :
    edgeTemp.append(edge)

def targetGetter() :
    return targetTemp

def edgeGetter() :
    return edgeTemp

def numGetter(file) :
    file = file.split('.')
    return file[0]

def run() :
    path = './blockImage/resize/metro'
    i = 0  # 결과영상 이름 찍는 용도로 쓰이는 변수임.
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':
                path += '/' + file
                print(path)
                # image 폴더내에 있는 모든 jpg 확장자 파일을 차례대로 읽어주는 반복문임
                # 만약 image 폴더내에 하위폴더가 있으면 그 하위폴더도 탐색함

                target = cv2.imread(path)  # 원본 영상을 가져옴
                target = removeNoise(target)  # 이미지 블러링하는 함수
                hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)  # 원본영상을 영상처리하기 위해 hsv로 변환
                #                extract_color(target, hsv_t)
                roi = trim(target)  # 영상분할을 위해서 쓸 샘플영상을 잘라오는 함수
                # roi라는 변수명은 backprojection 기법을 사용할 때 샘플영상을 다른 프로그래머들이 관습적으로 사용하는 변수이름임
                hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 샘플영상을 영상처리하기 위해 hsv로 변환

#                roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#                target2 = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
# 위에 2줄은 중간결과 보여줄 때 사용하려고 있는 변수임. 큰 의미없음
#                plt.subplot(3, 3, 1), plt.imshow(roi2), plt.title('sample'), plt.axis("off")
#                plt.subplot(3, 3, 2), plt.imshow(target2), plt.title('image'), plt.axis("off")
                dst = back_project(hsv_r, hsv_t)
                disc = convolution(dst)
                thresh = thresholding(disc, dst)
                res = merge(thresh, target)
                edge = edgeDetect(res)

                targetSetter(target)  # shapeDetect에 이미지를 넘겨주기 위한 함수
                edgeSetter(edge) #shapeDetect에 엣지를 넘겨주기 위한 함수
                numGetter(file)

#               plt.show()
#               res  = cv2.cvtColor(res ,cv2.COLOR_HSV2RGB)
#                cv2.imwrite("./resultTestImage/metro/target/" + str(i) + "target.jpg", target)
#                cv2.imwrite("./sample/" + str(i) + "sample.jpg", roi)
#                cv2.imwrite("./dst/" + str(i) + "dst.jpg", dst)
#                cv2.imwrite("./disc/" + str(i) + "disc.jpg", disc)
#                cv2.imwrite("./resultTestImage/metro/thresh/" + str(i) + "thresh.jpg", thresh)
#                cv2.imwrite("./res/" + str(i) + "res.jpg", res)
#                cv2.imwrite("./resultTestImage/metro/edge/" + str(i) + "edge.jpg", edge)

                path = './blockImage/resize/metro'
                i += 1

""" 파이에 실릴 함수
if __name__ == "__main__":
    # 수정된 시간을 활용하여 디렉토리 변화 유무를 감지하는 것임
    # 디렉토리내에 파일은 1개만 존재하며 활용된 영상은 바로 삭제하도록 하였음
    # 어떤 파일을 영상처리해야되는지에 대해서 코딩하려면 복잡하기 때문에 그렇게 한 것임
    path = './test' #임의의 디렉토리내에 영상이 촬영되었다고 가정함
    beforeTime = os.path.getmtime(path) #최초의 시간 설정
    while 1<2: #무한루프를 위한 설정
        currentTime = os.path.getmtime(path) # 현재 시점에서 수정된 날짜 감지
        print(currentTime)
        if beforeTime != currentTime : # 이전에 찍힌 시간과 현재 찍힌 시간이 다르다면 변화가 있었던 것
                target = cv2.imread("new.jpg")  # 원본 영상을 가져옴
                target = removeNoise(target)  # 이미지 블러링하는 함수
                hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)  # 원본영상을 영상처리하기 위해 hsv로 변환
                roi = trim(target)  # 영상분할을 위해서 쓸 샘플영상을 잘라오는 함수
                hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 샘플영상을 영상처리하기 위해 hsv로 변환

                dst = back_project(hsv_r, hsv_t)
                disc = convolution(dst)
                thresh = thresholding(disc, dst)
                res = merge(thresh, target)
                edge = edgeDetect(res)

                cv2.imwrite("NEWedge.jpg", edge) #임의의 파일에 엣지디텍트를 출력
                beforeTime = currentTime # 수정된 날짜 갱신
                if os.path.isfile("./test/new.jpg"):
                    os.remove("./test/new.jpg") # 이전에 있던 파일 삭제
"""

""" 파이에 실릴 메인함수
if __name__ == "__main__":
    path = './test'
    beforeTime = os.path.getmtime(path)
    while 1<2:
        currentTime = os.path.getmtime(path)
        print(currentTime)
        if beforeTime != currentTime :
            for root, dirs, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() == '.jpg':
                        target = cv2.imread(file)  # 원본 영상을 가져옴
                        target = removeNoise(target)  # 이미지 블러링하는 함수
                        hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)  # 원본영상을 영상처리하기 위해 hsv로 변환
                        roi = trim(target)  # 영상분할을 위해서 쓸 샘플영상을 잘라오는 함수
                        hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 샘플영상을 영상처리하기 위해 hsv로 변환

                        dst = back_project(hsv_r, hsv_t)
                        disc = convolution(dst)
                        thresh = thresholding(disc, dst)
                        res = merge(thresh, target)
                        edge = edgeDetect(res)

#                        cv2.imwrite("NEWsample.jpg", roi)
#                        cv2.imwrite("NEWdst.jpg", dst)
#                        cv2.imwrite("NEWdisc.jpg", disc)
#                        cv2.imwrite("NEWthresh.jpg", thresh)
#                        cv2.imwrite("NEWres.jpg", res)
                        cv2.imwrite("NEWedge.jpg", edge)
                        beforeTime = currentTime

                        
                print("완료")
"""


"""
if __name__ == "__main__":
    path = './blockImage/resize/metro'
    i = 0 # 결과영상 이름 찍는 용도로 쓰이는 변수임.
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg' :
                path += '/' + file
                print(path)
                # image 폴더내에 있는 모든 jpg 확장자 파일을 차례대로 읽어주는 반복문임
                # 만약 image 폴더내에 하위폴더가 있으면 그 하위폴더도 탐색함

                target = cv2.imread(path) # 원본 영상을 가져옴
                targetTemp = target # shapeDetect에 이미지를 넘겨주기 위한 별도의 저장
                target = removeNoise(target)  # 이미지 블러링하는 함수
                hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)  # 원본영상을 영상처리하기 위해 hsv로 변환
#                extract_color(target, hsv_t)
                roi = trim(target) # 영상분할을 위해서 쓸 샘플영상을 잘라오는 함수
                # roi라는 변수명은 backprojection 기법을 사용할 때 샘플영상을 다른 프로그래머들이 관습적으로 사용하는 변수이름임
                hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 샘플영상을 영상처리하기 위해 hsv로 변환


#                roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#                target2 = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                # 위에 2줄은 중간결과 보여줄 때 사용하려고 있는 변수임. 큰 의미없음
#                plt.subplot(3, 3, 1), plt.imshow(roi2), plt.title('sample'), plt.axis("off")
#                plt.subplot(3, 3, 2), plt.imshow(target2), plt.title('image'), plt.axis("off")
                dst = back_project(hsv_r, hsv_t)
                disc = convolution(dst)
                thresh = thresholding(disc, dst)
                res = merge(thresh, target)
                edge = edgeDetect(res)

                #plt.show()
                #res  = cv2.cvtColor(res ,cv2.COLOR_HSV2RGB)
#                cv2.imwrite("./resultTestImage/metro/target/" + str(i) + "target.jpg", target)
#                cv2.imwrite("./sample/" + str(i) + "sample.jpg", roi)
#                cv2.imwrite("./dst/" + str(i) + "dst.jpg", dst)
#                cv2.imwrite("./disc/" + str(i) + "disc.jpg", disc)
                cv2.imwrite("./resultTestImage/metro/thresh/" + str(i) + "thresh.jpg", thresh)
#                cv2.imwrite("./res/" + str(i) + "res.jpg", res)
                cv2.imwrite("./resultTestImage/metro/edge/" + str(i) + "edge.jpg", edge)

                path = './blockImage/resize/metro'
                i += 1
"""

""" 즉석 결과 확인용
if __name__ == "__main__":
    # function()

    roi = cv2.imread('sample.jpg')
    hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    target = cv2.imread('image.jpg')
    target = removeNoise(target)
    hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    target2 = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, 1), plt.imshow(roi2), plt.title('sample'), plt.axis("off")
    plt.subplot(3, 3, 2), plt.imshow(target2), plt.title('image'), plt.axis("off")

    dst = back_project(hsv_r, hsv_t)
    disc = convolution(dst)
    thresh = thresholding(disc, dst)
    res = merge(thresh, target)
    edge = edgeDetect(res)

    plt.show()
"""

"""
def function() :
    roi = cv2.imread('sample3.jpg')
    hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    target = cv2.imread('image3.jpg')

    target = cv2.GaussianBlur(target, (1, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    target = cv2.filter2D(target, -1, kernel)
    cv2.imshow('선명', target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv_t = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  #
    target2 = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)  #
    plt.subplot(3, 2, 1), plt.imshow(target2), plt.title('targetImage'), plt.axis("off")
    plt.subplot(3, 2, 2), plt.imshow(hsv_r), plt.title('HSV'), plt.axis("off")

    roihist = cv2.calcHist([hsv_r], [0, 1], None, [180, 256], [0, 180, 0, 256])

    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv_t], [0, 1], roihist, [0, 180, 0, 256], 1)
    plt.subplot(3, 2, 3), plt.imshow(dst, 'gray'), plt.title('backprojection'), plt.axis("off")

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)
    plt.subplot(3, 2, 4), plt.imshow(dst, 'gray'), plt.title('convolution'), plt.axis("off")

    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, disc, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, disc, iterations=2)
    plt.subplot(3, 2, 5), plt.imshow(thresh, 'gray'), plt.title('thresholding'), plt.axis("off")

    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(target, thresh)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 2, 6), plt.imshow(res), plt.title('result'), plt.axis("off")
    plt.show()

    edge = cv2.Canny(res, 100, 200)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
""" 색 추출을 위한 모듈
def extract_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of yellow lor in HSV
    lower_yellow = np.array([10,0,15])
    upper_yellow = np.array([50,255,180])    # VSH 순임
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_not(mask, mask)
    res = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow('frame', img)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = './image'
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg' :
                path += '/' + file
                print(path)
                trim(path)
                path = './image'
"""