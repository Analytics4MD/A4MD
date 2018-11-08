import numpy as np
import os
from glob import iglob
import subprocess
import freud

def analyze(types, points, box_points, step):
    #print('-----======= Python : analyze ========-------')
    print('box points',box_points);
    box = freud.box.Box(box_points[0],
        		box_points[1],
        		box_points[2],
        		box_points[3],
        		box_points[4],
        		box_points[5])
    voro = freud.voronoi.Voronoi(box, np.max(box.L)/2) 
    cells = voro.compute(box=box, positions=points).polytopes
    #print("In analyze: step: ",step)
    if step>=20000:
        voro.computeVolumes()
        frq,edges = np.histogram(voro.volumes,range=[0,0.5],bins=30)
        np.savetxt('voro_freq.txt',frq)
        np.savetxt('voro_edges.txt',edges)
        print('Number of voronoi cells',len(cells))
    return 0

if __name__ == "__main__":
    print('--------=========== Runnign analysis in python ===========-------------')

