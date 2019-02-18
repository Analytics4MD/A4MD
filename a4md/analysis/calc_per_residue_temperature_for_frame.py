import numpy as np
import os
import mdtraj as md
import time
from mdtraj.formats import PDBTrajectoryFile
from mdtraj import Trajectory
import pickle
import pbc as pbc
import trajectory_helper as traj_helper


nsteps = 1000
top = PDBTrajectoryFile(filename='reference.pdb').topology
atom_indices_by_resid = traj_helper.get_atoms_groups(top)
masses = traj_helper.get_masses(top)
indices = [a.index for a in top.atoms]
temperatures = []
bonds = [[bond.atom1.index,bond.atom2.index] for bond in top.bonds]
bond_dict = traj_helper.get_bond_dict(top, bonds)

fix_pbc = False

def analyze(types, xpoints, ypoints, zpoints, vel_x, vel_y, vel_z, box_points, step):
    print('-----======= Python : analyze ({})========-------'.format(step))
    # ---------============= analysis code goes here (start) ===========------------------- 
    points = np.vstack((xpoints, ypoints, zpoints)).T
    velocity = np.vstack((vel_x, vel_y, vel_z)).T
    axes = np.asarray(box_points)[:3]
    if fix_pbc:
        points = pbc.solve_positions_for_pbc(points, bonds, axes, bond_dict)
    per_res_temperature = [traj_helper.get_temperature(velocity[r],masses[r]) for r in atom_indices_by_resid]
    temperatures.append(per_res_temperature)
    # ---------============= analysis code goes here (end)   ===========------------------- 
    if step>=nsteps:
        print('------============ reached end of analysis ({}) ==========------------'.format(step))
        output = open('data.pkl', 'wb')
        pickle.dump(temperatures, output)
        output.close
        # This data can be read back as 
        # pkl_file = open('data.pkl', 'rb')
        # data1 = pickle.load(pkl_file)

    return 0

if __name__ == "__main__":
    print('--------=========== Running analysis in python ===========-------------')
