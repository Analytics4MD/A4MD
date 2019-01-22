import numpy as np
import os
from glob import iglob
import subprocess
import freud
import mdtraj as md
from timeit import default_timer as timer
from scipy import stats
import json
import time
from mdtraj.formats import PDBTrajectoryFile
from mdtraj import Trajectory
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.animation as manimation


def pbc_min_image(p1, p2, axes):
    dr = p1 - p2
    for i, p in enumerate(dr):
        if abs(dr[i]) > axes[i]*0.5:
            p2[i] = p2[i] + np.sign(dr[i])*axes[i] # Use dr to decide if we need to add or subtract axis
    return p2

def pbc_traslate(points, axes):
    '''Translates a group of points to the minimum image of first point in list
    points: list of poits
    axes: array, Lx, Ly, Lz
    We assume a tetragonal unit cell
    By default, all points will be translated into the minimum image
    of the first point, but any point can be used
    '''
    ref_point=points[0]
    # Add our min vector to our point
    min_image_cords = [pbc_min_image(ref_point, point, axes) for point in points[1:]] # Skip over the first point since its our ref point
    min_image_cords.insert(0, ref_point) # Don't forget to add the ref point into the list of points
    return np.asarray(min_image_cords)

def com(points, masses):
    #print(masses,points)
    weighted = masses[:,None]*points
    M = np.sum(masses)
    return np.sum(weighted,axis=0)/M#, M

def get_atoms_groups(topology, group_method='residue'):
    if group_method == 'residue':
        atom_indices = [[a.index for a in r.atoms] for r in topology.residues if r.is_protein]
    return np.asarray(atom_indices)

def get_masses(topology, group_method=None):
    if group_method == 'residue':
        masses = [np.asarray([a.element.mass for a in r.atoms]) for r in topology.residues if r.is_protein]
    elif group_method == None:
        masses = [a.element.mass for a in topology.atoms]
    else:
        raise NotImplementedError('get_masses not implemented')
    return np.asarray(masses)

def get_distances(atom_groups, xyzs, use_COM=True, masses=None):
    if use_COM and masses is not None:   
        coms_by_resid = [com(xyzs[r],masses[r]) for r in atom_groups]
        coords = np.asarray(coms_by_resid)
        #print(coords)
        distances = distance.cdist(coords, coords, 'euclidean')
        #distances = coms_by_resid
    else:
        raise ValueError('use_COM=False is not implemented')
    return distances


an_times = []
an_write_times = []
ddi = None
nsteps = None
with open('signac_statepoint.json', 'r') as f:
    sp = json.load(f)
    ddi = sp['stride']
    nsteps = sp['simulation_time']
top = PDBTrajectoryFile(filename='reference.pdb').topology
atom_indices_by_resid = get_atoms_groups(top)
masses = get_masses(top)
plt.figure()
fig, ax = plt.subplots()
ims = []

def analyze(types, xpoints,ypoints,zpoints, box_points, step):
    #print('-----======= Python : analyze ({})========-------'.format(step))
    #print('box points',box_points);
    #print("");
    t=timer() 
    # ---------============= analysis code goes here (start) ===========------------------- 
    points = np.vstack((xpoints,ypoints,zpoints)).T
    min_img_cords = pbc_traslate(points,np.asarray(box_points[:3]))
    dist_matrix = get_distances(atom_indices_by_resid,min_img_cords,masses=masses)
    im = ax.imshow(dist_matrix, interpolation='nearest')
    ims.append([im])
    plt.draw()
    # ---------============= analysis code goes here (end)   ===========------------------- 
    an_time = timer()-t
    an_times.append(an_time)

    t=timer()
    # ---------============= analysis output code goes here (start) ===========------------------- 

    # ---------============= analysis output code goes here (end)   ===========------------------- 
    an_write_time = timer()-t
    an_write_times.append(an_write_time)    
    
    if step>=nsteps:
        print('------============ reached end of analysis ({}) ==========------------'.format(step))
        ani = manimation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save('dis.html')
        with open('signac_job_document.json', 'r') as f:
            job_document = json.load(f)

        job_document['analysis_time_s'] = np.sum(an_times)
        job_document['analysis_time_s_sem'] = stats.sem(an_times)
        job_document['analysis_output_time_s'] = np.sum(an_write_times)
        job_document['analysis_output_time_s_sem'] = stats.sem(an_write_times)

        with open('signac_job_document.json', 'w') as f:
            f.write(json.dumps(job_document))

    return 0

if __name__ == "__main__":
    print('--------=========== Runnign analysis in python ===========-------------')

