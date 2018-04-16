import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

def getMax(cnt):
    cnt = np.reshape(cnt,(-1,2))
    return np.array([np.max(cnt[:,1]),np.max(cnt[:,0])],dtype=np.int)


def getMin(cnt):
    cnt = np.reshape(cnt,(-1,2))
    return np.array([np.min(cnt[:,1]),np.min(cnt[:,0])],dtype=np.int)

def findLimts(skel):
    limitPoint = []
    contours = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnts in contours:
        if type(cnts) == list:
            for cnt in cnts:
                limitPoint.append(getMax(cnt))
                limitPoint.append(getMin(cnt))
        else:
            cnt = cnts
            limitPoint.append(getMax(cnt))
            limitPoint.append(getMin(cnt))
    return limitPoint

def findKeypoints(skel,limitPoint):
    endPoint = []
    forkPoint = []
    whitepoint = np.asarray(np.where(skel==255)).T
    limitPoint = np.asarray(limitPoint)
    # [whitepoint[0][0], whitepoint[1][0]
    for xy in whitepoint:
        # TODO: si pertenece xy a los limites no usar
        if not np.any(limitPoint==xy):
            num = np.count_nonzero(skel[xy[0]-1:xy[0]+2,xy[1]-1:xy[1]+2])-1
            if num==1:
                endPoint.append(xy)
            if num > 2:
                forkPoint.append(xy)

    return endPoint,forkPoint

def drawPoint(image,lst_point,color):
    out = image.copy()
    for xy in lst_point:
        out[xy[0],xy[1],:] = np.asarray(color)
    return out

if __name__=="__main__":
    img = cv2.imread('huellasFVC04/101_2.tif', 0)
    img = 255*np.ones(img.shape)-img
    img = img.astype(np.uint8)
    img_logical = np.logical_and(img>100,img>101)
    original = img.copy()



    # Invert the horse image


    skeleton = skeletonize(img_logical)

    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True)

    ax = axes.ravel()

    skel = 255*skeleton.astype(np.uint8)
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('cv2 ', fontsize=20)

    ax[1].imshow(skel, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

    limitPoint = findLimts(skel)
    endPoint,forkPoint = findKeypoints(skel,limitPoint)
    out = cv2.cvtColor(skel,cv2.COLOR_GRAY2RGB)
    #
    out = drawPoint(out,endPoint,(255,0,0))
    out = drawPoint(out,forkPoint,(0,255,0))
    # out = drawPoint(out,limitPoint,(0,0,255))


    # plt.hist(original.ravel(),256,[0,256]); plt.show()

    # plt.imshow(skel,cmap="gray"); plt.show()

    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(skel, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('skeleton ', fontsize=20)

    ax[1].imshow(out)
    ax[1].axis('off')
    ax[1].set_title('limits', fontsize=20)


    fig.tight_layout()
    plt.show()

    cv2.imshow("out",out)
    cv2.waitKey()



    # cv2.imshow("skel", skel)
    # cv2.imshow("original", original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()