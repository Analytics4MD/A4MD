import numpy as np
from scipy.spatial import distance
import freud.msd as freud_msd


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

def get_distances(atom_groups, xyzs, use_COM=False, masses=None):
    if use_COM and masses is not None:
        coms_by_resid = [com(xyzs[r],masses[r]) for r in atom_groups]
        coords = np.asarray(coms_by_resid)
        #print(coords)
        distances = distance.cdist(coords, coords, 'sqeuclidean')
        #distances = coms_by_resid
    elif masses == None:
        atom_groups = np.asarray(atom_groups)
        coords = xyzs[atom_groups]
        distances = distance.cdist(coords, coords, 'sqeuclidean')
    else:
        raise ValueError('use_COM=False is not implemented')
    return distances

def get_atoms_groups(topology, group_method='residue'):
    if group_method == 'residue':
        atom_indices = [[a.index for a in r.atoms] for r in topology.residues if r.is_protein]
    elif group_method == 'alpha_carbon':
        atom_indices = [atom.index for atom in topology.atoms if (atom.name == 'CA')] 
    return np.asarray(atom_indices)   

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def get_segments(topology, segment_length, include_last=True):
    CA_indices = get_atoms_groups(topology, group_method='alpha_carbon')
    print('CA_indices',CA_indices)
    if segment_length > len(CA_indices):
        print('WARNING: number of alpha carbon atoms is less than segment length.')
        print('Number of CA atoms:',len(CA_indices))
        print('Segment length assigned:',segment_length)
        print('Choosing {} as the segment length'.format(len(CA_indices)))
        segment_length = len(CA_indices)
    segment_ids = chunks(CA_indices, segment_length)
    sids = [segid for segid in segment_ids]
    if len(sids[-1]) != segment_length:
        if include_last:  
            # use a backward view for the last segment. Ok with overlap.
            sids[-1] = CA_indices[-segment_length*2:-segment_length]
    return sids

def get_bond_dict(top, bonds):
    bond_dict = {atom.index:[] for atom in top.atoms}
    for bond in bonds:
        if bond[1] not in bond_dict[bond[0]]:
            bond_dict[bond[0]].append(bond[1])
        if bond[0] not in bond_dict[bond[1]]:
            bond_dict[bond[1]].append(bond[0])
    return bond_dict

def get_temperature(velocities, masses):
    '''
        Calculates the instantaneous kinetic temperature of a given set of atoms
        whose velocities and masses are given.
        It is important to note that the average kinetic energy used here is 
        limited to the translational kinetic energy of the molecules. 
        That is, they are treated as point masses and no account is made of 
        internal degrees of freedom such as molecular rotation and vibration.

        $KE_{avg} = \frac{1}{2} \overline{m v^2} = \frac{3}{2} k_BT$

        k_B, the Botlzmann constant is taken as 1.38066\times10^{-23} J/K

        Parameters:
        velocities: ndarray, shape=(Number of atoms,Number of dimensions). Values are expected to be in SI units (m/s)
        masses: ndarray, values expected to be in SI units (kg)
        output: temperature value in SI unit (Kelvin)
    '''
    kB = 1.38066e-23
    d = velocities.shape[1] #  getting the dimension
    T = np.mean(masses*velocities**2)/(d*kB)
    return T
    
def get_rmsd(points, ref_points, method='direct'):
    if method == 'direct':
        direct_msd = freud_msd.MSD(mode=method)
        allpos = np.stack((ref_points, points),axis=0)
        msd_val = direct_msd.compute(allpos)
        rmsd = np.sqrt(msd_val.msd)
    elif method == 'window':
        window_msd = freud_msd.MSD(mode=method)
        allpos = np.stack((ref_points, points),axis=0)
        msd_val = window_msd.compute(allpos)
        rmsd = np.sqrt(msd_val.msd)
    else:
        raise NotImplementedError('rmsd method {} is not implemeted'.format(method))
    return rmsd
