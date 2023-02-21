import numpy as np

def direct(types, posx, posy, posz, box, step):
    print(f'forward::direct:: writing forwarded cords to file')
    file_name='log.cords_from_intermediator'
    with open(file_name,'a') as f:
         f.write(f'X:{posx}\n')
         f.write(f'Y:{posy}\n')
         f.write(f'Z:{posz}\n')
    #write to a file
    return types, posx, posy, posz, box, step
