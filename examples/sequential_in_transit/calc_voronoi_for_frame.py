#import MDAnalysis
import numpy as np
import os
from glob import iglob
import subprocess
import freud

#def get_box():
#    with open('lj.data') as fp:  
#        line = fp.readline()
#        print(line)
#        cnt = 1
#        while line:
#            if 'xhi' in line:
#                values = line.split(' ')
#                xlow = float(values[0])
#                xhi = float(values[1])
#            if 'yhi' in line:
#                values = line.split(' ')
#                ylow = float(values[0])
#                yhi = float(values[1])
#            if 'zhi' in line:
#                values = line.split(' ')
#                zlow = float(values[0])
#                zhi = float(values[1])
#                break
#            line = fp.readline()
#    box = freud.box.Box(xhi,yhi,zhi)
#    print('Made box:',box)
#    return box

def analyze(types, points, box_points):
    print('-----======= Python : analyze ========-------')
    print('box points',box_points);
    box = freud.box.Box(box_points[0],
			box_points[1],
 			box_points[2],
			box_points[3],
			box_points[4],
			box_points[5])
    voro = freud.voronoi.Voronoi(box, np.max(box.L)/2) 
    voro.compute(box=box, positions=points)

if __name__ == "__main__":
    print('--------=========== Runnign analysis in python ===========-------------')
# This will return absolute paths
#if os.path.isfile('lj.data') and os.path.isfile('output.xyz'): 
#    u = MDAnalysis.Universe('output.xyz')
#    box = get_box()
#    all_atoms = u.select_atoms('all')
#    for frame in u.trajectory:
#        points = all_atoms.positions
#        voro = freud.voronoi.Voronoi(box, np.max(box.L)/2) 
#        voro.compute(box=box, positions=points)
