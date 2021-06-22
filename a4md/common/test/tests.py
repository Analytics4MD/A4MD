import numpy as np
import time
from pya4md import *

def my_function():
    print("Hello from a function")

def add(i, j):
    return i + j

def extract_frame(fileName, position, nAtoms):
    """
    Keep extract frame information in trajectory file
    
    Parameters
    ------------
    fileName  : string 
            Trajectory file path
    postition : integer
            Position to seek frame
    nAtoms    : integer
            Configurable knobs: number of atoms

    Returns
    ------------
    num : int
            Number of atoms in frame
    CA  : list of tuples
            Set of atom positions

    """
    file = open(fileName)
    try:
        file.seek(position)
    except:
        file.seek(0)
    
    x_cords = []
    y_cords = []
    z_cords = []
    types = []
    #CA = []
    num = 0
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.1)
            continue
        # TODO: set timer for exit

        elif line[:4]=="ATOM":
            if line[12:16]==" CA ":
                alternate_location = line[16]
                if alternate_location==' ' or alternate_location=='A':
                    #CA.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                    x_cords.append(float(line[30:38]))
                    y_cords.append(float(line[38:46]))
                    z_cords.append(float(line[46:54]))
                    types.append(int(line[22:26]))
                    num += 1

        elif line[:6]=="ENDMDL":
            curr_natoms = len(x_cords)
            if nAtoms > 0 and curr_natoms != nAtoms:
                # Concatenate list of atoms
                x_cords = (x_cords * (nAtoms // curr_natoms + 1))[:nAtoms]
                y_cords = (y_cords * (nAtoms // curr_natoms + 1))[:nAtoms]
                z_cords = (z_cords * (nAtoms // curr_natoms + 1))[:nAtoms]
                types = (types * (nAtoms // curr_natoms + 1))[:nAtoms]    
            break

        elif line[:3]=="END":
            # Repeat searching from beginning  
            file.seek(0)
            num = -1
            #break 

    lp = file.tell()
    chunk = MDChunk(1, 1, types, x_cords, y_cords, z_cords, 1, 2, 3, 4, 5, 6)
    return chunk

def mdchunk_to_mdchunk(chunk):
    return chunk

def mdchunk_to_cvchunk(chunk):
    id = chunk.getChunkID()
    cv_vals = chunk.getXPositions()
    return CVChunk(id, cv_vals)


