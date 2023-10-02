#!/usr/bin/python

import numpy as np
import time

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
    
    x_cords = np.random.uniform(low=0, high=2, size=nAtoms).tolist()
    y_cords = np.random.uniform(low=0, high=2, size=nAtoms).tolist()
    z_cords = np.random.uniform(low=0, high=2, size=nAtoms).tolist()
    types = np.random.randint(low=0, high=10, size=nAtoms).tolist()
    print(f'load::extract_frame:: writing ingested cords to file')
    file_name='log.cords_from_ingester'
    with open(file_name,'a') as f:
         f.write(f'X:{x_cords}\n')
         f.write(f'Y:{y_cords}\n')
         f.write(f'Z:{z_cords}\n')
    return types, x_cords, y_cords, z_cords, [None, None, None, None, None, None], None, 0

