from flow import FlowProject
import flow
import subprocess
import environment
import numpy as np
import os
from glob import iglob
import freud
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

def diskstats_parse(dev=None):
    file_path = '/proc/diskstats'
    result = {}

    # ref: http://lxr.osuosl.org/source/Documentation/iostats.txt
    columns_disk = ['m', 'mm', 'dev', 'reads', 'rd_mrg', 'rd_sectors',
                    'ms_reading', 'writes', 'wr_mrg', 'wr_sectors',
                    'ms_writing', 'cur_ios', 'ms_doing_io', 'ms_weighted']

    columns_partition = ['m', 'mm', 'dev', 'reads', 'rd_sectors', 'writes', 'wr_sectors']

    lines = open(file_path, 'r').readlines()
    for line in lines:
        if line == '': continue
        split = line.split()
        if len(split) == len(columns_disk):
            columns = columns_disk
        elif len(split) == len(columns_partition):
            columns = columns_partition
        else:
            # No match
            continue

        data = dict(zip(columns, split))
        if dev != None and dev != data['dev']:
            continue
        for key in data:
            if key != 'dev':
                data[key] = int(data[key])
        result[data['dev']] = data

    return result

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
TARGET=NONE TOTAL_STEPS={}\n".\
                       format(job.sp.data_dump_interval,
                              job.sp.simulation_time))
        elif job.sp.job_type == 'plumed_sequential':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
TARGET=calc_voronoi_for_frame PYTHON_FUNCTION=analyze TOTAL_STEPS={}\n".\
                       format(job.sp.data_dump_interval,
                              job.sp.simulation_time))
        elif job.sp.job_type == 'plumed_conc_NO':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
                       TARGET=a4md TOTAL_STEPS={}\n".\
                       format(job.sp.data_dump_interval,
                              job.sp.simulation_time))

    if job.sp.job_type == 'plumed_conc_NO':
        with open(job.fn('dataspaces.conf'), 'w') as file:
            if job.sp.job_type != 'traditional':
                file.write("## Config file for DataSpaces\n")
                file.write("ndim = 1\n")
                file.write("dims = 100000\n")
                file.write("max_versions = 1000\n")
                file.write("max_readers = 1\n")
                file.write("lock_type = 2\n")
                file.write("hash_version = 1\n")

    copyfile('analysis_codes/calc_voronoi_for_frame.py',job.fn('calc_voronoi_for_frame.py'))
    copyfile('checkio.sh',job.fn('checkio.sh'))

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
        if 'analysis_output_time' in job.document:
            tt += job.document['analysis_output_time']
        else:
            raise ValueError('analysis_output_time not found in job document')

    else:
        job.document['ete_analysis_time'] = get_ete_analysis_time(job)
        if 'analysis_time' in job.document: # analysis time is written from the analysis script
            tt = job.document['ete_analysis_time']-job.document['analysis_time']
        else:
            raise ValueError('analysis_time is not in the job document.')
        if 'analysis_output_time' in job.document:
            tt -= job.document['analysis_output_time']
        else:
            raise ValueError('analysis_output_time not found in job document')

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
    import time
    with job, open("generate_stdout.log", "w+") as generate_stdout,\
              open('iostats.log','w+') as iostat_out,\
              open('retriever.log','w+') as retriever_out,\
              open('server.log','w+') as ds_server_out:
        job_command = ['bash','checkio.sh']
        generate_iostats = subprocess.Popen(job_command, stdout=iostat_out, stderr=iostat_out)
        time.sleep(10)

        if job.sp.job_type == 'plumed_conc_NO':
            job_command = ['mpirun','-n','1','dataspaces_server','-s','1','-c','2']
            generate_ds_server = subprocess.Popen(job_command, stdout=ds_server_out, stderr=ds_server_out)
            # Give some time for the servers to load and startup
            time.sleep(3) #  wait server to fill up the conf file

            job_command = ['mpirun','-n','1','retriever','calc_voronoi_for_frame','analyze',str(job.sp.simulation_time),str(job.sp.data_dump_interval)]
            generate_retriever = subprocess.Popen(job_command, stdout=retriever_out, stderr=retriever_out)

        job_command = ['mpirun','-n',str(job.sp.NPROCS),'lmp_mpi','-i','in.lj']
        print("Executing job command:", job_command)
        start = timer()
        generate = subprocess.Popen(job_command, stdout=generate_stdout, stderr=generate_stdout)
        generate.wait()
        if job.sp.job_type == 'plumed_conc_NO':
            generate_retriever.wait()

        t = timer() - start
        job.document['ete_simulation_time'] = t

        if job.sp.job_type == 'plumed_conc_NO':
            generate_ds_server.kill()

        time.sleep(10)
        generate_iostats.kill()
    

@A4MDProject.operation
@A4MDProject.pre(simulated)
@A4MDProject.post(analyzed)
@flow.directives(np=8)
def analyze_job(job):
    if job.sp.job_type == 'traditional':
        import mdtraj as md 
        with job:
            if 'read_frames_time' not in job.document:
                read_frame_times = []
                start = timer()
                import analysis_codes.calc_voronoi_for_frame as calc_voro
                traj_ext = job.sp.output_type
                traj_file = 'output.{}'.format(traj_ext)
                top_file = 'top_L_{}.pdb'.format(job.sp.L)
                if job.isfile(top_file) and job.isfile(traj_file): 
                    if job.sp.output_type =='dcd':
                        traj = md.load_dcd(traj_file,top=top_file)
                    elif job.sp.output_type == 'xyz':
                        traj = md.load_xyz(traj_file,top=top_file)
                    else:
                        raise ValueError('Unrecognized output_type: {}'.format(job.sp.output_type))
                    if len(traj)>0:
                        print('found',len(traj),'frames in',traj_file)
                        box_L = [job.sp.L, job.sp.L, job.sp.L]#traj[0].unitcell_lengths[0]*10 # mul by 10 to compensate for mdtraj dividing by 10
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
                    else:
                        raise ValueError('trajectory file {} does not contain any frames.'.format(traj_file))
                else:
                    raise ValueError('top file ({}) or traj file ({}) not found in job ({})'.format(top_file,traj_file,job))

                job.document['read_frames_time']=np.sum(read_frame_times)        

    job.document['transfer_time'] = get_transfer_time(job)
    job.document['modify_time'] = get_modify_time(job)
    job.document['simulation_time'] = get_simulation_time(job)


if __name__ == '__main__':
    A4MDProject().main()
