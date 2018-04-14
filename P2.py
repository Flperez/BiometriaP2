import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def Skeletonization(img):
    img = cv2.GaussianBlur(img, (5, 5), 0.2)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def findKeypoints(skel):
    endPoint = []
    forkPoint = []
    whitepoint = np.asarray(np.where(skel==255)).T

    # [whitepoint[0][0], whitepoint[1][0]
    for xy in whitepoint:
        num = np.count_nonzero(skel[xy[0]-1:xy[0]+1,xy[1]-1:xy[1]+1])-1
        if num==1:
            endPoint.append(xy)
        if num > 2:
            forkPoint.append(xy)

    return endPoint,forkPoint

def drawPoint(image,lst_point,color):
    for xy in lst_point:
        cv2.circle(image,(xy[1],xy[0]),1,color)
    return out

if __name__=="__main__":
    img = cv2.imread('huellasFVC04/101_2.tif', 0)
    img = 255*np.ones(img.shape)-img
    img = img.astype(np.uint8)
    original = img.copy()

    skel = Skeletonization(img)
    # skeleton = skeletonize(img)
    endPoint,forkPoint = findKeypoints(skel)
    out = cv2.cvtColor(skel,cv2.COLOR_GRAY2RGB)

    out = drawPoint(out,endPoint,(255,0,0))
    out = drawPoint(out,forkPoint,(0,255,0))


    # plt.hist(original.ravel(),256,[0,256]); plt.show()

    plt.imshow(skel,cmap="gray"); plt.show()
    plt.imshow(out); plt.show()
    # cv2.imshow("skel", skel)
    # cv2.imshow("original", original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()