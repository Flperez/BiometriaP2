import numpy as np
import argparse,os

def evaluatePoint(path_gt,path_result, offset=5):
    with open(path_gt,'r') as file:
        data = list(file)
        lst_gt = [list(map(int,datum.rstrip().split(' '))) for datum in data]

    with open(path_result, 'r') as file:
        data = list(file)
        lst_result = [list(map(int,datum.rstrip().split(' '))) for datum in data]

    correct = 0
    for result in lst_result:
        # comprobamos si el punto se encuentra cerca de algun punto del gt
        for gt in lst_gt:
            if gt[0]-offset<result[0]<gt[0]+offset and gt[1]-offset<result[1]<gt[1]+offset:
                correct+=1
                break
    return correct,len(lst_result)-correct


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True,
                    help="ruta a la carpeta de gt: <name>_end.txt and <name>_fork.txt")
    ap.add_argument("--result", required=True,
                    help="ruta a la carpeta de resultados: <name>_end.txt and <name>_fork.txt")
    args = vars(ap.parse_args())
    path_gt = args['gt']
    path_result = args['result']

    ## Endpoint ##
    name = path_gt.split('/')[-1]
    endcorrect,endfalse = evaluatePoint(os.path.join(path_gt,name+"_end.txt"),
                                          os.path.join(path_result,name+"_end.txt"))
    ## Fork point ##
    forkcorrect, forkfalse = evaluatePoint(os.path.join(path_gt, name + "_fork.txt"),
                                         os.path.join(path_result, name + "_fork.txt"))

    print("---------",name,"---------")
    print("endPoints: corrects:%d false positives:%d".format(endcorrect,endfalse))
    print("forkPoints: corrects:%d false positives:%d".format(forkcorrect,forkfalse))
