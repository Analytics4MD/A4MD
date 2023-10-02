import a4md.analysis.trajectory_helper
import numpy as np


def move_bonded_atoms(central_atom, positions, axes, bond_dict):
    for bonded_atom in bond_dict[central_atom]:
        atom1_posn = positions[central_atom]
        atom2_posn = positions[bonded_atom]
        sep_vec = np.array(atom1_posn) - np.array(atom2_posn)
        moved = False
        for axis, value in enumerate(sep_vec):
            if value > axes[axis] / 2.0:
                positions[bonded_atom][axis] += axes[axis]
                moved = True
            if value < -axes[axis] / 2.0:
                positions[bonded_atom][axis] -= axes[axis]
                moved = True
        if moved:
            positions = move_bonded_atoms(bonded_atom, positions, axes, bond_dict)
    return positions

def check_bonds(positions, bonds, axes, bond_dict):
    for bond in bonds:
        posn1 = np.array(positions[bond[0]])
        posn2 = np.array(positions[bond[1]])
        separation = posn1-posn2
        if any(separation >= axes / 2.0):
            print("Periodic bond found:",bond,
                "because separation =",separation,">=",axes / 2.0)
            positions = move_bonded_atoms(bond[1], positions, axes, bond_dict)
    return positions

def solve_positions_for_pbc(positions, bonds, axes, bond_dict):
    '''
    positions: array of xyz positions [[x1, y1, z1], [x1,y2,z2],...[xn,y,z,]]
    bonds: array of bonded atom indices [[b1_1,b1_2],[b2_1,b2_2],...[bn_1,bn_2]]
    axes: array box edges (assumed orthogonal box)
    bond_dict: dictionary of bonds where keys are atom indices and values are bonded atom indices
    '''
    positions = check_bonds(positions,bonds,axes,bond_dict)
    return positions
