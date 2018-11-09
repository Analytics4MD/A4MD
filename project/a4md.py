from flow import FlowProject
import flow
import subprocess
import environment
import numpy as np
import os
from glob import iglob
import freud
import mdtraj as md 



class A4MDProject(FlowProject):
    pass


def initialized(job):
    return job.isfile('in.lj')

def simulated(job):
    if job.sp.job_type == 'traditional':
        return job.isfile('output.dcd')
    else:
        return analyzed(job)

def analyzed(job):
    #print('in job',job,job.isfile('voro_freq.txt'))
    return job.isfile('voro_freq.txt')

@A4MDProject.operation
@A4MDProject.post(initialized)
def initialize(job):
    import create_lammps_input as lmp
    from shutil import copyfile
    source_file = 'files/top_L_{}.pdb'.format(job.sp.L)
    dest_file = job.fn('top_L_{}.pdb'.format(job.sp.L))
    print('Copying {} to {}'.format(source_file,dest_file))
    copyfile(source_file,dest_file)
    copyfile('analysis_codes/calc_voronoi_for_frame.py',job.fn('calc_voronoi_for_frame.py'))
    with job:
        lmp.create_lammps_script(job)
    with open(job.fn('plumed.dat'), 'w') as file:
        file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} TARGET=LICHENS\n".format(job.sp.data_dump_interval))

@A4MDProject.operation
@A4MDProject.pre(initialized)
@A4MDProject.post(simulated)
@flow.cmd
def simulate(job):
    return "cd {job.ws} && mpirun -n {job.sp.NPROCS} lmp_mpi -i in.lj"

@A4MDProject.operation
@A4MDProject.pre(simulated)
@A4MDProject.post(analyzed)
def analyze_job(job):
    if job.sp.job_type == 'traditional':
        with job:
            import calc_voronoi_for_frame as calc_voro
            traj_ext = 'dcd'
            traj_file = 'output.{}'.format(traj_ext)
            top_file = 'top_L_{}.pdb'.format(job.sp.L)
            if job.isfile(top_file) and job.isfile(traj_file): 
                traj = md.load_dcd(traj_file,top=top_file)
                if len(traj)>0:
                    box_L = traj[0].unitcell_lengths[0]*10 # mul by 10 to compensate for mdtraj dividing by 10
                    print(box_L, type(box_L))
                    box_points=np.append(box_L,[0, 0, 0])
                    print("my box points are ", box_points)
                    print("total frames",traj.n_frames)
                    for frame in range(traj.n_frames):
                        points = traj.xyz[frame]*10
                        dummy=[]
                        calc_voro.analyze(dummy, points, box_points, frame*job.sp.data_dump_interval) 


if __name__ == '__main__':
    A4MDProject().main()
