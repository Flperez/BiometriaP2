import argparse
from tools import *
import os
from natsort import natsorted


def main(input,output,visu=False):
    names = natsorted(os.listdir(input))
    for name in names:
        # 1.- Cargamos las imagenes
        img, img_invert, img_logical = load_image(os.path.join(input,name))

        # 2.- Esqueletizamos la huella
        skel_logical, skel_image = calcSkel(img_logical)

        # 3.- Buscamos los puntos en los que no deberiamos aplicar la busqueda de minucias
        mask = findLimts(skel_image)

        # 4.- Buscamos los puntos finales y de bifurcacion
        endPoint, forkPoint = findKeypoints(skel_image,mask)



        if visu:
            # plot2images(img_invert,img_logical)
            # plot2images(img_logical,skel_logical)
            # plot2images(skel_logical,skel_image)
            out = cv2.cvtColor(img_invert.copy(),cv2.COLOR_GRAY2RGB)
            # out = drawPoint(out, endPoint, (255, 0, 0))
            # out = drawPoint(out, forkPoint, (0, 255, 0))

            out = drawCircle(out, endPoint, (255, 0, 0))
            out = drawCircle(out, forkPoint, (0, 255, 0))

            plot2images(skel_logical,out)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="ruta a la carpeta de entrada de imagenes")
    ap.add_argument("--output", required=False,
                    help="ruta a la carpeta de salida de imagenes")
    args = vars(ap.parse_args())
    input = args['input']
    output = args['output']
    main(input,output,True)



