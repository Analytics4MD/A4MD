import numpy as np

def mdchunk_to_mdchunk(types, posx, posy, posz, box, step):
    return types, posx, posy, posz, box, step

def mdchunk_to_cvchunk(types, posx, posy, posz, box, step):
    return posx