import cv2,os
import numpy as np
import argparse
from tools.tools import *

# mouse callback function
endPoint = []
forkPoint = []

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img_invert,(x,y),4,(255,0,0),-1)
        print('endPoint: ',x,y)
        endPoint.append([x,y])
    if event == cv2.EVENT_RBUTTONDBLCLK:
        cv2.circle(img_invert,(x,y),4,(0,255,0),-1)
        print('forkPoint: ',x,y)
        forkPoint.append([x,y])

def save(input,output):
    if not os.path.exists(output):
        os.mkdir(output)
    name = input.split('/')[-1].split('.')[0]
    with open(os.path.join(output,name+"_end.txt"),'w') as file:
        for x,y in endPoint:
            file.write("%d %d\n"%(int(0.5*x),int(0.5*y)))

    with open(os.path.join(output, name + "_fork.txt"), 'w') as file:
        for x, y in forkPoint:
            file.write("%d %d\n"% (int(x*.5), int(0.5*y)))


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="ruta a la imagen de entrada")
    ap.add_argument("--output", required=True,
                    help="ruta a la carpeta de salida del gt creado")
    args = vars(ap.parse_args())
    input = args['input']
    output = args['output']
    # Create a black image, a window and bind the function to window
    _, img_invert, _ = load_image(input,1)
    img_invert = cv2.resize(img_invert,None,fx=2,fy=2)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img_invert)
        key=cv2.waitKey(1)
        if key == 113: # key=q
            break
        if key == 115: # key = s
            print('saving result')
            save(input,output)
    cv2.destroyAllWindows()
