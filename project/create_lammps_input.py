

def create_lammps_script(job, file_name='in.lj'):
    with open(file_name, 'w') as f:
        f.write('#  define units\n')
        f.write('units       lj\n')
        f.write('\n')
        f.write('#  specify periodic boundary conditions\n')
        f.write('boundary p p p\n')
        f.write('\n')
        f.write('#  define atom_style\n')
        f.write('#  full covers everything\n')
        f.write('atom_style  atomic \n')
        f.write('\n')
        f.write('#  define simulation volume \n')
        f.write('#  If I want N = 512 atoms \n')
        f.write('#  and I want a density of rho = 0.5 atoms/lj-sigma^3\n')
        f.write('#  Then I can determine the size of a cube by \n')
        f.write('#  size = (N/rho)^(1/3)\n')
        f.write('#  10.0793684\n')
        f.write('region      boxid block 0.0 {} 0.0 {} 0.0 {}\n'.format(job.sp.L,job.sp.L,job.sp.L))
        f.write('\n')
        f.write('create_box  1 boxid\n')
        f.write('\n')
        f.write('#  specify initial positions of atoms\n')
        f.write('#  sc = simple cubic\n')
        f.write('#  0.5 = density in lj units\n')
        f.write('lattice     sc 0.45\n')
        f.write('\n')
        f.write('#  place atoms of type 1 in boxid\n')
        f.write('create_atoms    1 box\n')
        f.write('\n')
        f.write('#   define mass of atom type 1\n')
        f.write('mass        1 1.0\n')
        f.write('\n')
        f.write('#  specify initial velocity of atoms\n')
        f.write('#  group = all\n')
        f.write('#  reduced temperature is T = 1.0 = lj-eps/kb \n')
        f.write('#  seed for random number generator\n')
        f.write('#  distribution is gaussian (e.g. Maxwell-Boltzmann)\n')
        f.write('velocity    all create 1.0 87287 dist gaussian\n')
        f.write('\n')
        f.write('#  specify interaction potential\n')
        f.write('#  pairwise interaction via the Lennard-Jones potential with a cut-off at 2.5 lj-sigma\n')
        f.write('pair_style  lj/cut 2.5\n')
        f.write('\n')
        f.write('#  specify parameters between atoms of type 1 with an atom of type 1\n')
        f.write('#  epsilon = 1.0, sigma = 1.0, cutoff = 2.5\n')
        f.write('pair_coeff  1 1 1.0 1.0 2.5\n')
        f.write('\n')
        f.write('# add long-range tail correction\n')
        f.write('pair_modify tail yes\n')
        f.write('\n')
        f.write('#  specify parameters for neighbor list \n')
        f.write('#  rnbr = rcut + 0.3\n')
        f.write('neighbor    0.3 bin\n')
        f.write('\n')
        f.write('#  specify thermodynamic properties to be output\n')
        f.write('#  pe = potential energy\n')
        f.write('#  ke = kinetic energy\n')
        f.write('#  etotal = pe + ke\n')
        f.write('#  temp = temperature\n')
        f.write('#  press = pressure\n')
        f.write('#  density = number density\n')
        f.write('#  output every thousand steps\n')
        f.write('#  norm = normalize by # of atoms (yes or no)\n')
        f.write('thermo_style custom step pe ke etotal temp press density\n')
        f.write('\n')
        f.write('# report instantaneous thermo values every 100 steps\n')
        f.write('thermo 100\n')
        f.write('\n')
        f.write('# normalize thermo properties by number of atoms (yes or no)\n')
        f.write('thermo_modify norm no\n')
        f.write('\n')
        f.write('#  specify ensemble\n')
        f.write('#  fixid = 1\n')
        f.write('#  atoms = all\n')
        f.write('#  ensemble = nve or nvt\n')
        f.write('fix     1 all nve\n')
        f.write('\n')
        f.write('#  define time step\n')
        f.write('timestep 0.0005\n')
        f.write('\n')
        f.write('# run 1000 steps in the NVE ensemble\n')

        f.write('# (this equilibrates positions)\n')
        f.write('run 1000\n')
        f.write('\n')
        f.write('#  stop fix with given fixid\n')
        f.write('#  fixid = 1\n')
        f.write('unfix 1\n')
        f.write('\n')
        f.write('#  specify ensemble\n')
        f.write('#  fixid = 2\n')
        f.write('#  atoms = all\n')
        f.write('#  ensemble = nvt\n')
        f.write('#  temp = temperature\n')
        f.write('#  initial temperature = 1.0\n')
        f.write('#  final temperature = 1.0\n')
        f.write('#  thermostat controller gain = 0.1 (units of time, bigger is less tight control)\n')
        f.write('fix     2 all nvt temp {} {} 0.5\n'.format(job.sp.T,job.sp.T))
        f.write('\n')
        f.write('# run 1000 steps in the NVT ensemble\n')
        f.write('# (this equilibrates thermostat)\n')
        f.write('run     1000\n')
        f.write('\n')
        f.write('#   save configurations\n')
        f.write('#   dumpid = 1\n')
        f.write('#   all atoms\n')
        f.write('#   atomic symbol is Ar\n')
        f.write('#   save positions every 100 steps\n')
        f.write('#   filename = output.xyz\n')
        f.write('#\n')
        f.write('\n')
        f.write('reset_timestep 0\n')
        f.write('write_data lj.data\n')
        f.write('\n')

        if job.sp.job_type == 'traditional':
            #if 'output_type' not in job.sp:
            #    job.sp['output_type']='dcd'
            if 'output_type' in job.sp:
                if job.sp.output_type == 'dcd':
                    f.write('dump    1       all dcd {} output.dcd\n'.format(job.sp.data_dump_interval))
                elif job.sp.output_type == 'xyz':
                    f.write('dump    1       all xyz {} output.xyz\n'.format(job.sp.data_dump_interval))
            f.write('#dump_modify 1 element Ar\n')
        #elif 'plumed' in job.sp.job_type:
        f.write('fix  3 all plumed plumedfile plumed.dat outfile plumed.out \n')

        f.write('\n')
        f.write('# run 1000 more steps in the NVT ensemble\n')
        f.write('# (this is data production, from which configurations are saved) \n')
        f.write('log log.prod\n')
        f.write('run  {}\n'.format(job.sp.simulation_time))
