import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects



########## Paso 1 ##########
def load_image(path):
    # Load image
    img = cv2.imread(path, 0)

    # invert image
    img_invert = 255 * np.ones(img.shape) - img
    img_invert = img_invert.astype(np.uint8)

    # Logical image
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    th3 =  cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img_logical = 255 * np.ones(img.shape) - th3
    img_logical = np.logical_and(img_logical > 170, img_logical > 171)
    return img,img_invert,img_logical


########## Paso 2 ##########
def calcSkel(img_logical):
    skel_logical = skeletonize(img_logical)
    skel_image = 255 * skel_logical.astype(np.uint8)
    return skel_logical, skel_image


########## Paso 3 ##########
def findLimts(skel):
    #TODO: eliminar motas
    mask = np.ones(skel.shape,dtype=bool)
    # Buscamos el punto min y max de cada fila
    whitepoint = np.asarray(np.where(skel==255)).T
    fil_unique,indexes = np.unique(whitepoint[:,0],return_index=True)
    lst_limits = np.array([],dtype=np.int)
    for x in range(indexes.shape[0]-1):
        points = whitepoint[indexes[x]:indexes[x+1],:]
        maximum = np.max(points[:,1])
        minimum = np.min(points[:,1])
        lst_limits = np.append(lst_limits,np.array([fil_unique[x],maximum]))
        lst_limits = np.append(lst_limits,np.array([fil_unique[x],minimum]))
    lst_limits = np.reshape(lst_limits,(-1,2))
    for limit in lst_limits:
        mask[limit[0], limit[1]]=False

    return mask

########## Paso 4 ##########

def findKeypoints(skel,mask):
    endPoint = []
    forkPoint = []
    whitepoint = np.asarray(np.where(skel==255)).T
    for xy in whitepoint:
        if mask[xy[0], xy[1]]:
            num = np.count_nonzero(skel[xy[0]-1:xy[0]+2,xy[1]-1:xy[1]+2])-1
            if num==1:
                endPoint.append(xy)
            if num > 2:
                forkPoint.append(xy)

    return endPoint,forkPoint

########## Paso 5 ##########
def drawPoint(image,lst_point,color):
    out = image.copy()
    for xy in lst_point:
        out[xy[0],xy[1],:] = np.asarray(color)
    return out

def drawCircle(image,lst_point,color):
    out = image.copy()
    for xy in lst_point:
        cv2.circle(out,(xy[1],xy[0]),1,color,2)
    return out




def plot2images(img1,img2,title1='img1',title2='img2',visu=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img1, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title(title1, fontsize=20)

    ax[1].imshow(img2, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title(title2, fontsize=20)

    fig.tight_layout()
    if visu:
        plt.show()
    return fig

def plot4images(images,titles):
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()