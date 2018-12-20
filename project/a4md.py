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

def processed(job):
    return 'ete_workflow_time_s' in job.document and \
           'analysis_time_s' in job.document

def post_processed(job):
    #print('in job',job,job.isfile('voro_freq.txt'))
    return 'ete_workflow_time_s' in job.document and \
           'simulation_time_s' in job.document and \
           'analysis_time_s' in job.document and \
           'plumed_overhead_time_s' in job.document and \
           'stage_in_time_s' in job.document and \
           'stage_out_time_s' in job.document

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
        if job.sp.job_type == 'plumed_sequential':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
TARGET=py PYTHON_MODULE=calc_voronoi_for_frame PYTHON_FUNCTION=analyze \
TOTAL_STEPS={}\n".\
                       format(job.sp.data_dump_interval,
                              job.sp.simulation_time))
        elif job.sp.job_type == 'plumed_ds_sequential':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
TARGET=py PYTHON_MODULE=calc_voronoi_for_frame PYTHON_FUNCTION=analyze \
TOTAL_STEPS={} STAGE_DATA_IN=dataspaces\n".\
                       format(job.sp.data_dump_interval,
                              job.sp.simulation_time))
        elif job.sp.job_type == 'plumed_ds_concurrent':
                    file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
TARGET=a4md TOTAL_STEPS={} STAGE_DATA_IN=dataspaces\n".\
                               format(job.sp.data_dump_interval,
                                      job.sp.simulation_time))

    if 'plumed_ds_' in job.sp.job_type:
        with open(job.fn('dataspaces.conf'), 'w') as file:
                file.write("## Config file for DataSpaces\n")
                file.write("ndim = 1\n")
                file.write("dims = 100000\n")
                file.write("max_versions = 1000\n")
                file.write("max_readers = 1\n")
                if job.sp.job_type == 'plumed_ds_sequential':
                     file.write("lock_type = 1\n") #  We are going to write first and read.
                elif job.sp.job_type == 'plumed_ds_concurrent':
                     file.write("lock_type = 3\n")
                file.write("hash_version = 1\n")

    copyfile('analysis_codes/calc_voronoi_for_frame.py',job.fn('calc_voronoi_for_frame.py'))
    copyfile('checkio.sh',job.fn('checkio.sh'))

    with job:
        lmp.create_lammps_script(job)


def get_ete_plumed_time(job):
    ete_plumed_time = None
    with job, open("plumed.out", "r") as log:
            labels = ['total_dispatch_action_time_ms']
            for line in log:
                values = line.split(':')
                if any(label in values[1] for label in labels):
                    values = line.split(':')
                    ete_plumed_time = float(values[2])
                    break
    if ete_plumed_time is None:
        raise ValueError('Could not find total_dispatch_action_time_ms in plumed.out')
    return ete_plumed_time

def get_read_plumed_data_time_s(job):
    time = None
    with job, open("plumed.out", "r") as log:
            labels = ['total_read_plumed_data_time_ms']
            for line in log:
                values = line.split(':')
                if any(label in values[1] for label in labels):
                    values = line.split(':')
                    time = float(values[2])*1e-3
                    break
    if time is None:
        raise ValueError('Could not find total_read_plumed_data_time_ms in plumed.out')
    return time

def get_dispatch_action_time_s(job):
    dispatch_action_time = None
    with job, open("plumed.out", "r") as log:
        labels = ['total_dispatch_action_time_ms']
        for line in log:
            if labels[0] in line:
                values = line.split(':')
                dispatch_action_time = float(values[2])*1e-3
    if dispatch_action_time is None:
        raise ValueError('Could not find total_dispatch_action_time_ms in plumed.out')
    return dispatch_action_time

def get_stage_in_time_s(job):
    time = None
    if job.sp.job_type == 'plumed_ds_concurrent' or job.sp.job_type == 'plumed_ds_sequential':
        stage_in_time = None
        with job, open("generate_stdout.log", "r") as log:
                labels = ['total_data_write_time_ms']
                for line in log:
                    if labels[0] in line:
                        values = line.split(':')
                        stage_in_time = float(values[1])*1e-3

        if stage_in_time is None:
            raise ValueError('Could not find total_data_write_time_ms in generate_stdout.log')
        time = stage_in_time
    elif job.sp.job_type == 'plumed_sequential':
        time = 0.0 #  No stage in time for plumed sequential since we do not stage data
    elif job.sp.job_type == 'traditional':
        output_time_labels = ['Output']#'Pair','Neigh']#,'Comm','Other']#,'Modify','Ouput']
        with job, open("log.prod", "r") as log:
            for line in log:
                values = line.split('|')
                if len(values)== 6 and any(label in values[0] for label in output_time_labels):
                    values = line.split('|')
                    time = float(values[2]) # Adding up times for all the analysis_time_labels
                    break
    else:
        raise ValueError('Please implement get_stage_in_time_s for {}'.format(job.sp.job_type))

    if time is None:
        raise ValueError('stage in time not found for {}'.format(job.sp.job_type))

    return time

def get_total_retrieve_time_s(job):
    time = None
    with job, open("plumed.out", "r") as log:
        labels = ['total_retriever_time_ms']
        for line in log:
            if labels[0] in line:
                values = line.split(':')
                time = float(values[2])*1e-3
    if time is None:
        raise ValueError('Could not find total_retriever_time_ms in plumed.out')
    return time

def get_plumed_overhead_time_s(job):
    time = 0.0
    if job.sp.job_type == 'plumed_ds_concurrent':
        dispatch_action_time = get_dispatch_action_time_s(job)
        stage_in_time = get_stage_in_time_s(job) 
        time = dispatch_action_time - stage_in_time
    elif job.sp.job_type == 'plumed_ds_sequential':
        dispatch_action_time = get_dispatch_action_time_s(job)
        stage_in_time = get_stage_in_time_s(job) 
        retrieve_time = get_total_retrieve_time_s(job)
        time = dispatch_action_time - stage_in_time - retrieve_time
    elif job.sp.job_type == 'plumed_sequential':
        dispatch_action_time = get_dispatch_action_time_s(job)
        retrieve_time = get_total_retrieve_time_s(job)
        time = dispatch_action_time - retrieve_time
    elif job.sp.job_type == 'traditional':
        time = 0.0
    else:
        raise ValueError('Please implement get_plumed_overhead_time_s for {}'.format(job.sp.job_type))

    return time


def get_stage_out_time_s(job):
    time = 0.0
    if 'plumed_ds_' in job.sp.job_type:
        if job.sp.job_type == 'plumed_ds_concurrent':
            log_file = 'retriever.log'
        elif job.sp.job_type == 'plumed_ds_sequential':
            log_file = 'generate_stdout.log'
        stage_out_time = None
        with job, open(log_file, "r") as log:
                labels = ['total_data_read_time_ms']
                for line in log:
                    if labels[0] in line:
                        values = line.split(':')
                        stage_out_time = float(values[1])*1e-3

        if stage_out_time is None:
            raise ValueError('Could not find total_data_read_time_ms in {}'.format(log_file))
        time = stage_out_time
    elif job.sp.job_type == 'plumed_sequential':
        time = 0.0
    elif job.sp.job_type == 'traditional':
        time = job.document['read_frames_time']
    else:
        raise ValueError('Please implement get_stage_out_time_s for {}'.format(job.sp.job_type))

    return time


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
        job.document['ete_plumed_time'] = get_ete_plumed_time(job)
        if 'analysis_time' in job.document: # analysis time is written from the analysis script
            tt = job.document['ete_plumed_time']-job.document['analysis_time']
        else:
            raise ValueError('analysis_time is not in the job document.')
    return tt

def get_overhead_time(job):
    ot = 0.0
    if job.sp.job_type == 'traditional':
        output_time_labels = ['Ouput']#'Pair','Neigh']#,'Comm','Other']#,'Modify','Ouput']
        with job, open("log.prod", "r") as log:
            for line in log:
                values = line.split('|')
                if len(values)== 6 and any(label in values[0] for label in output_time_labels):
                    values = line.split('|')
                    ot = float(values[2]) # Adding up times for all the analysis_time_labels
                    break
        if 'read_frames_time' in job.document:
            ot += job.document['read_frames_time']
        else:
            raise ValueError('read_frames_time not found in job document')
        if 'analysis_output_time' in job.document:
            ot += job.document['analysis_output_time']
        else:
            raise ValueError('analysis_output_time not found in job document')

    else:
        ot = job.document['ete_workflow_time_s'] - job.document['simulation_time'] - job.document['analysis_time']
    
    return ot

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


def get_simulation_time_s(job):
    st = 0.0
    if job.sp.job_type == 'traditional':
        modify_time_labels = ['Pair','Neigh','Comm','Other']#,'Modify','Ouput']
        with job, open("log.prod", "r") as log:
            for line in log:
                values = line.split('|')
                if len(values)== 6 and any(label in values[0] for label in modify_time_labels):
                    values = line.split('|')
                    st += float(values[2]) # Adding up times for all the analysis_time_labels
    else:
        with job, open("plumed.out", "r") as log:
            labels = ['total_simulation_time_ms']
            for line in log:
                values = line.split(':')
                if any(label in values[1] for label in labels):
                    values = line.split(':')
                    st = float(values[2])*0.001 # Adding up times for all the analysis_time_labels
                    break
    return st


def traditional_analysis(job):
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
                        calc_voro.analyze(dummy, points[:,0], points[:,1],points[:,2], box_points, frame*job.sp.data_dump_interval,) 
                else:
                    raise ValueError('trajectory file {} does not contain any frames.'.format(traj_file))
            else:
                raise ValueError('top file ({}) or traj file ({}) not found in job ({})'.format(top_file,traj_file,job))

            job.document['read_frames_time']=np.sum(read_frame_times)

@A4MDProject.operation
@A4MDProject.pre(initialized)
@A4MDProject.post(processed)
@flow.directives(np=18)
#@flow.cmd
def process(job):
    import time
    with job, open("generate_stdout.log", "w+") as generate_stdout,\
              open('iostats.log','w+') as iostat_out,\
              open('retriever.log','w+') as retriever_out,\
              open('server.log','w+') as ds_server_out:
        job_command = ['bash','checkio.sh']
        generate_iostats = subprocess.Popen(job_command, stdout=iostat_out, stderr=iostat_out)
        time.sleep(0)

        if 'plumed_ds_' in job.sp.job_type:
            if job.sp.job_type == 'plumed_ds_concurrent':
                job_command = ['mpirun','-n','1','dataspaces_server','-s','1','-c','2']
            elif job.sp.job_type == 'plumed_ds_sequential':
                job_command = ['mpirun','-n','1','dataspaces_server','-s','1','-c','1']
            generate_ds_server = subprocess.Popen(job_command, stdout=ds_server_out, stderr=ds_server_out, shell=False)
            # Give some time for the servers to load and startup
            time.sleep(3) #  wait server to fill up the conf file

        if job.sp.job_type == 'plumed_ds_concurrent':
            one_extra_step = 1 # plumed sends the initial frame as well
            n_frames = int(job.sp.simulation_time/job.sp.data_dump_interval) + one_extra_step
            job_command = ['mpirun','-n','1','retriever','calc_voronoi_for_frame','analyze', str(n_frames)]
            generate_retriever = subprocess.Popen(job_command, stdout=retriever_out, stderr=retriever_out, shell=False)

        job_command = ['mpirun','-n',str(job.sp.NPROCS),'lmp_mpi','-i','in.lj']
        #job_command = ['mpirun -n {} lmp_mpi -i in.lj'.format(job.sp.NPROCS)]
        print("Executing job command:", job_command)
        start = timer()
        generate = subprocess.Popen(job_command, stdout=generate_stdout, stderr=generate_stdout, shell=False)
        generate.wait()
        if job.sp.job_type == 'plumed_ds_concurrent':
            generate_retriever.wait()

        if job.sp.job_type == 'traditional':
            start_analysis = timer()
            traditional_analysis(job)
            job.document['ete_analysis_time_s'] = timer() - start_analysis
     
        t = timer() - start
        job.document['ete_workflow_time_s'] = t

        if 'plumed_ds_' in job.sp.job_type:
            generate_ds_server.kill()

        time.sleep(0)
        generate_iostats.kill()
    

@A4MDProject.operation
@A4MDProject.pre(processed)
@A4MDProject.post(post_processed)
@flow.directives(np=8)
def post_process(job):
    #job.document['transfer_time'] = get_transfer_time(job)
    #job.document['modify_time'] = get_modify_time(job)
    job.document['simulation_time_s'] = get_simulation_time_s(job)
    job.document['overhead_time_s'] = job.document['ete_workflow_time_s'] - job.document['simulation_time_s'] - job.document['analysis_time_s']
    job.document['plumed_overhead_time_s'] = get_plumed_overhead_time_s(job)
    job.document['stage_in_time_s'] = get_stage_in_time_s(job)
    job.document['stage_out_time_s'] = get_stage_out_time_s(job)

if __name__ == '__main__':
    A4MDProject().main()
