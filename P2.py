import argparse
from tools import *
import os
from natsort import natsorted


def main(input,output,visu=False):
    names = natsorted(os.listdir(input))
    for name in names:
        print("\nImage: ",name)
        # 1.- Cargamos las imagenes
        img, img_invert, img_logical = load_image(os.path.join(input,name))

        # 2.- Esqueletizamos la huella
        skel_logical, skel_image = calcSkel(img_logical)

        # 3.- Buscamos los puntos en los que no deberiamos aplicar la busqueda de minucias
        mask = findLimts(skel_image)

        # 4.- Buscamos los puntos finales y de bifurcacion
        endPoint, forkPoint = findKeypoints(skel_image,mask)
        print("\tNumber of endpoints: ",len(endPoint))
        print("\tNumber of forkpoints: ",len(forkPoint))


        # 5.- Visualizamos o guardamos las imagenes
        if output or visu:
            fig1 = plot2images(img_invert, img_logical, 'huella', 'binaria', visu)
            fig2 = plot2images(img_logical, skel_logical, 'binaria', 'esqueletizacion', visu)

            out = cv2.cvtColor(img_invert.copy(), cv2.COLOR_GRAY2RGB)
            out = drawCircle(out, endPoint, (255, 0, 0))
            out = drawCircle(out, forkPoint, (0, 255, 0))
            fig3 = plot2images(skel_logical, out, 'binaria', 'resultado', visu)
            if output:
                path_image = os.path.join(output,name.split('.')[0])
                if not os.path.isdir(path_image):
                    os.mkdir(path_image)
                fig1.savefig(os.path.join(path_image,"huella_vs_binaria.png"))
                fig2.savefig(os.path.join(path_image,"binaria_vs_skel.png"))
                fig3.savefig(os.path.join(path_image,"skel_vs_result.png"))





if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="ruta a la carpeta de entrada de imagenes")
    ap.add_argument("--output", required=False,
                    help="ruta a la carpeta de salida de imagenes")
    ap.add_argument("--visu",action="store_true",help="visualizar los pasos seguidos")
    args = vars(ap.parse_args())
    input = args['input']
    output = args['output']
    visu = ap.parse_args().visu
    main(input,output,visu)