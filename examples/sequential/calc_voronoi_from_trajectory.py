import mdtraj as md 
import numpy as np
import os
from glob import iglob
import subprocess
import freud
import sys

L=15
if len(sys.argv) > 1:
    L = sys.argv[1]
    print('Using L={} for calc_voronoi script in python'.format(L))
# This will return absolute paths
traj_ext = 'dcd'
traj_file = 'output.{}'.format(traj_ext)
top_file = 'top_L_{}.pdb'.format(L)
if os.path.isfile(top_file) and os.path.isfile(traj_file): 
    traj = md.load_dcd(traj_file,top=top_file)
    if len(traj)>0:
        box_L = traj[0].unitcell_lengths[0]
        box = freud.box.Box(box_L[0],box_L[1],box_L[2])
        for frame in range(traj.n_frames):
            points = traj.xyz[frame]
            voro = freud.voronoi.Voronoi(box, np.max(box.L)/2) 
            cells = voro.compute(box=box, positions=points).polytopes
        print('Number of voronoi cells',len(cells)) 
