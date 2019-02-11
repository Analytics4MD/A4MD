import numpy as np

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

def get_bond_dict(traj, bonds):
    bond_dict = {atom.index:[] for atom in traj.topology.atoms}
    for bond in bonds:
        if bond[1] not in bond_dict[bond[0]]:
            bond_dict[bond[0]].append(bond[1])
        if bond[0] not in bond_dict[bond[1]]:
            bond_dict[bond[1]].append(bond[0])
    return bond_dict
