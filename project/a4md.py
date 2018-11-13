from flow import FlowProject
import flow
import subprocess
import environment
import numpy as np
import os
from glob import iglob
import freud
import mdtraj as md 
from timeit import default_timer as timer
from scipy import stats


class A4MDProject(FlowProject):
    pass

def initialized(job):
    return job.isfile('in.lj')

def simulated(job):
    return 'ete_simulation_time' in job.document

def analyzed(job):
    #print('in job',job,job.isfile('voro_freq.txt'))
    return 'transfer_time' in job.document and \
           'simulation_time' in job.document and \
           'modify_time' in job.document

@A4MDProject.operation
@A4MDProject.post(initialized)
def initialize(job):
    import create_lammps_input as lmp
    from shutil import copyfile
    if job.sp.job_type == 'traditional':
        source_file = 'files/top_L_{}.pdb'.format(job.sp.L)
        dest_file = job.fn('top_L_{}.pdb'.format(job.sp.L))
        print('Copying {} to {}'.format(source_file,dest_file))
        copyfile(source_file,dest_file)
    #elif 'plumed' in job.sp.job_type:
    with open(job.fn('plumed.dat'), 'w') as file:
        if job.sp.job_type == 'traditional':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
                       TARGET=NONE\n".\
                       format(job.sp.data_dump_interval))
        else:
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
                       TARGET=PYTHON PYTHON_MODULE=calc_voronoi_for_frame\n".\
                       format(job.sp.data_dump_interval))


    copyfile('analysis_codes/calc_voronoi_for_frame.py',job.fn('calc_voronoi_for_frame.py'))
    with job:
        lmp.create_lammps_script(job)


def get_ete_analysis_time(job):
    ete_analysis_time = None
    with job, open("generate_stdout.log", "r") as log:
        for line in log:
            if "DISPATCH_UPDATE_TIME_rank_0" in line:
                ete_analysis_time = float(line.split(':')[1])/1000.0
                break
    if ete_analysis_time is None:
        raise ValueError('Could not find DISPATCH_UPDATE_TIME_rank_0 in generate_stdout.log')
    return ete_analysis_time

def get_transfer_time(job):
    tt = 0
    if job.sp.job_type == 'traditional':
        output_time_labels = ['Ouput']#'Pair','Neigh']#,'Comm','Other']#,'Modify','Ouput']
        with job, open("log.prod", "r") as log:
            for line in log:
                values = line.split('|')
                if len(values)== 6 and any(label in values[0] for label in output_time_labels):
                    values = line.split('|')
                    tt = float(values[2]) # Adding up times for all the analysis_time_labels
                    break
        if 'read_frames_time' in job.document:
            tt += job.document['read_frames_time']
        else:
            raise ValueError('read_frames_time not found in job document')
    else:
        job.document['ete_analysis_time'] = get_ete_analysis_time(job)
        if 'analysis_time' in job.document: # analysis time is written from the analysis script
            tt = job.document['ete_analysis_time']-job.document['analysis_time']
        else:
            raise ValueError('analysis_time is not in the job document.')
    return tt


def get_modify_time(job):
    mt = 0
    modify_time_labels = ['Modify']#'Pair','Neigh']#,'Comm','Other']#,'Modify','Ouput']
    with job, open("log.prod", "r") as log:
        for line in log:
            values = line.split('|')
            if len(values)== 6 and any(label in values[0] for label in modify_time_labels):
                values = line.split('|')
                mt = float(values[2]) # Adding up times for all the analysis_time_labels
    return mt


def get_simulation_time(job):
    st = 0
    modify_time_labels = ['Pair','Neigh','Comm','Other']#,'Modify','Ouput']
    with job, open("log.prod", "r") as log:
        for line in log:
            values = line.split('|')
            if len(values)== 6 and any(label in values[0] for label in modify_time_labels):
                values = line.split('|')
                st += float(values[2]) # Adding up times for all the analysis_time_labels
    return st


@A4MDProject.operation
@A4MDProject.pre(initialized)
@A4MDProject.post(simulated)
@flow.directives(np=18)
#@flow.cmd
def simulate(job):
    with job, open("generate_stdout.log", "w+") as generate_stdout:
        job_command = ['mpirun','-n',str(job.sp.NPROCS),'lmp_mpi','-i','in.lj']
        print("Executing job command:", job_command)
        start = timer()
        generate = subprocess.Popen(job_command, stdout=generate_stdout, stderr=generate_stdout)
        generate.wait()
        t = timer() - start
        job.document['ete_simulation_time'] = t
    

@A4MDProject.operation
@A4MDProject.pre(simulated)
@A4MDProject.post(analyzed)
@flow.directives(np=8)
def analyze_job(job):
    if job.sp.job_type == 'traditional':
        with job:
            if 'read_frames_time' not in job.document:
                read_frame_times = []
                start = timer()
                import analysis_codes.calc_voronoi_for_frame as calc_voro
                traj_ext = 'dcd'
                traj_file = 'output.{}'.format(traj_ext)
                top_file = 'top_L_{}.pdb'.format(job.sp.L)
                if job.isfile(top_file) and job.isfile(traj_file): 
                    traj = md.load_dcd(traj_file,top=top_file)
                    if len(traj)>0:
                        box_L = traj[0].unitcell_lengths[0]*10 # mul by 10 to compensate for mdtraj dividing by 10
                        #print(box_L, type(box_L))
                        box_points=np.append(box_L,[0, 0, 0])
                        #print("my box points are ", box_points)
                        #print("total frames",traj.n_frames)
                        read_frame_times.append(timer()-start)
                        for frame in range(traj.n_frames):
                            start = timer()
                            points = traj.xyz[frame]*10
                            dummy=[0]*len(points)
                            read_frame_times.append(timer()-start)
                            calc_voro.analyze(dummy, points, box_points, frame*job.sp.data_dump_interval) 
                job.document['read_frames_time']=np.sum(read_frame_times)        

    job.document['transfer_time'] = get_transfer_time(job)
    job.document['modify_time'] = get_modify_time(job)
    job.document['simulation_time'] = get_simulation_time(job)

if __name__ == '__main__':
    A4MDProject().main()
