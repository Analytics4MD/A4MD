import numpy as np

def analyze(types, posx, posy, posz, box, step):
    print(isinstance(types,(list,np.ndarray)),isinstance(types[0],int))
    print(type(types[0]))
    if isinstance(types,(list,np.ndarray)) and isinstance(types[0],int) and \
       isinstance(posx,list) and isinstance(posx[0],float) and \
       isinstance(box,list) and isinstance(box[0],float) and isinstance(step,int):
        return 0
    else:
        return -1

