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


def get_dispatch_action_time_s(job):
    dispatch_action_time = None
    with job, open("md.log", "r") as log:
        labels = ['total_dispatch_action_time_ms']
        for line in log:
            if labels[0] in line:
                values = line.split(':')
                dispatch_action_time = float(values[2])*1e-3
                break
    if dispatch_action_time is None:
        raise ValueError('Could not find total_dispatch_action_time_ms in plumed.log')
    return dispatch_action_time


def get_total_data_read_time_s(job):
    time = None
    if job.sp.job_type == 'plumed_ds_concurrent':
        with job, open("retriever.log", "r") as log:
                labels = ['total_data_read_time_ms']
                for line in log:
                    if labels[0] in line:
                        values = line.split(':')
                        time = float(values[1])*1e-3
                        job.document['total_data_read_time_s'] = time
                        break
    else:
        raise ValueError('Please implement get_total_data_read_time_s for {}'.format(job.sp.job_type))

    if time is None:
        raise ValueError('data read time not found for {}'.format(job.sp.job_type))

    return time


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
                    time = float(values[2])
                    break
    else:
        raise ValueError('Please implement get_stage_in_time_s for {}'.format(job.sp.job_type))

    if time is None:
        raise ValueError('stage in time not found for {}'.format(job.sp.job_type))

    return time


def get_total_pyanalyzer_time_s(job):
    time = None
    with job, open("md.log", "r") as log:
        labels = ['total_pyanalyzer_time_ms']
        for line in log:
            if labels[0] in line:
                values = line.split(':')
                time = float(values[2])*1e-3
                break
    if time is None:
        raise ValueError('Could not find total_pyanalyzer_time_ms in md.log')
    return time


def get_data_management_time_s(job):
    time = 0.0
    if job.sp.job_type == 'plumed_ds_concurrent':
        dispatch_action_time = get_dispatch_action_time_s(job)
        total_data_read_time_s = get_total_data_read_time_s(job)
        print('-----=========== total read time : {}'.format(total_data_read_time_s))
        time = dispatch_action_time + total_data_read_time_s
    elif job.sp.job_type == 'plumed_ds_sequential':
        dispatch_action_time = get_dispatch_action_time_s(job)
        pyanalyzer_time = get_total_pyanalyzer_time_s(job)
        time = dispatch_action_time - pyanalyzer_time
    elif job.sp.job_type == 'plumed_sequential':
        dispatch_action_time = get_dispatch_action_time_s(job)
        pyanalyzer_time = get_total_pyanalyzer_time_s(job)
        time = dispatch_action_time - pyanalyzer_time
    elif job.sp.job_type == 'traditional':
        time = get_transfer_time(job)
    else:
        raise ValueError('Please implement get_plumed_overhead_time_s for {}'.format(job.sp.job_type))

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
        pyanalyzer_time = get_total_pyanalyzer_time_s(job)
        time = dispatch_action_time - stage_in_time - pyanalyzer_time
    elif job.sp.job_type == 'plumed_sequential':
        dispatch_action_time = get_dispatch_action_time_s(job)
        pyanalyzer_time = get_total_pyanalyzer_time_s(job)
        time = dispatch_action_time - pyanalyzer_time
    elif job.sp.job_type == 'traditional':
        time = 0.0
    else:
        raise ValueError('Please implement get_plumed_overhead_time_s for {}'.format(job.sp.job_type))

    return time


def get_transfer_time(job):
    tt = 0
    if job.sp.job_type == 'traditional':
        output_time_labels = ['Write']#'Pair','Neigh']#,'Comm','Other']#,'Modify','Ouput']
        with job, open("md.log", "r") as log:
            for line in log:
                values = line.split()
                if len(values)== 8 and values[0] == 'Write' and values[1] == 'traj.':#any(label in values[0] for label in output_time_labels):
                    tt = float(values[5])
                    break
        if 'read_frames_time' in job.document:
            tt += job.document['read_frames_time']
        else:
            raise ValueError('read_frames_time not found in job document')
        if 'analysis_output_time_s' in job.document:
            tt += job.document['analysis_output_time_s']
        else:
            raise ValueError('analysis_output_time_s not found in job document')

    else:
        raise ValueError('get_transfer_time is only implemented for traditional workflow.')
    return tt


def get_simulation_time_s(job):
    st = 0.0
    if job.sp.job_type == 'traditional':
        with job, open("md.log", "r") as log:
            for line in log:
                if 'Time:' in line:
                    values = line.split()
                    st = float(values[2])
                    break
        st = st-get_transfer_time(job)
    else:
        with job, open("md.log", "r") as log:
            labels = ['total_simulation_time_ms']
            for line in log:
                if any(label in line for label in labels):
                    values = line.split(':')
                    st = float(values[2])*0.001
                    break
    return st


def traditional_analysis(job):
    import mdtraj as md 
    with job:
        if 'read_frames_time' not in job.document:
            read_frame_times = []
            start = timer()
            import analysis_codes.calc_dist_matrix_for_frame as calc_dist
            traj_ext = job.sp.output_type
            traj_file = 'traj_comp.{}'.format(traj_ext)
            top_file = 'reference.pdb'
            if job.isfile(top_file) and job.isfile(traj_file): 
                traj = md.load(traj_file,top=top_file)
                read_frame_times.append(timer()-start)

                if len(traj)>0:
                    print('found',len(traj),'frames in',traj_file)
                    #print("my box points are ", box_points)
                    #print("total frames",traj.n_frames)
                    for frame in range(traj.n_frames):
                        box_points=[traj.unitcell_lengths[0][0],traj.unitcell_lengths[0][1],traj.unitcell_lengths[0][2],0,0,0]
                        points = traj.xyz[frame]*10
                        dummy=[0]*len(points)
                        calc_dist.analyze(dummy, points[:,0], points[:,1],points[:,2], box_points, frame*job.sp.data_dump_interval) 
                else:
                    raise ValueError('trajectory file {} does not contain any frames.'.format(traj_file))
            else:
                raise ValueError('top file ({}) or traj file ({}) not found in job ({})'.format(top_file,traj_file,job))

            job.document['read_frames_time'] = np.sum(read_frame_times)
            if 'analysis_time_s' not in job.document:
                raise ValueError('analysis_time_s is not found in job document!')

def barplot(df,variable,data_dump_interval,ylabel,show_n_largest=10,multiplier=1,logscale=True):
    import matplotlib.pyplot as plt
    df_iter = df
    fig1, ax1 = plt.subplots()

    df_sorted = df_iter.sort_values(variable,ascending=False).head(show_n_largest)
    num_functions = df_sorted['Function'].nunique()
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0., 1., num_functions))
    print(len(df_sorted['Function_short']))
    ax1.bar(np.arange(len(df_sorted['Function_short'])),
            df_sorted[variable]*multiplier)
    if logscale:
        ax1.set_yscale('log')
    ax1.set_xticks(np.arange(len(df_sorted['Function_short'])))
    ax1.set_xticklabels(df_sorted['Function_short'], rotation = 45, ha='right', zorder=100,fontsize=10)
    print(df_sorted['Function_short'].values)
    ax1.set_xlabel('Function Name')
    ax1.set_ylabel(ylabel)
    plt.title('{} (Stride: {})'.format(variable,data_dump_interval), fontsize=10)
    plt.savefig('retriever_profile.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches='tight',# pad_inches=0.1,
                frameon=None, metadata=None)


class A4MDProject(FlowProject):
    pass

def initialized(job):
    return job.isfile('topol.tpr')

def processed(job):
    return 'total_time_s' in job.document and \
           'analysis_time_s' in job.document

def post_processed(job):
    #print('in job',job,job.isfile('voro_freq.txt'))
    if 'tau_profiling' in job.sp and job.sp.tau_profiling and job.sp.job_type=='plumed_ds_concurrent':
        profiled = job.isfile('profile.tsv') and job.isfile('retriever_profile.png')
    else:
        profiled = True
    return profiled and \
           'total_time_s' in job.document and \
           'simulation_time_s' in job.document and \
           'analysis_time_s' in job.document and \
           'data_management_time_s' in job.document


@A4MDProject.operation
@A4MDProject.post(initialized)
def initialize(job):
    import create_gmx_input as gmx_in
    from shutil import copyfile
    with open(job.fn('plumed.dat'), 'w') as file:
        if job.sp.job_type == 'plumed_sequential':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
TARGET=py PYTHON_MODULE=calc_dist_matrix_for_frame PYTHON_FUNCTION=analyze \
TOTAL_STEPS={}\n".\
                       format(job.sp.data_dump_interval,
                              job.sp.simulation_time))
        elif job.sp.job_type == 'plumed_ds_sequential':
            file.write("p: DISPATCHATOMS ATOMS=@mdatoms STRIDE={} \
TARGET=py PYTHON_MODULE=calc_dist_matrix_for_frame PYTHON_FUNCTION=analyze \
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

    copyfile('files/gltph_gmx/start_conf.gro',job.fn('start_conf.gro'))
    copyfile('files/gltph_gmx/remake_tpr.sh',job.fn('remake_tpr.sh'))
    copyfile('files/gltph_gmx/top/topol.top',job.fn('topol.top'))
    copyfile('files/gltph_gmx/index.ndx',job.fn('index.ndx'))
    copyfile('files/gltph_gmx/start_state.cpt',job.fn('start_state.cpt'))
    with job:
        gmx_in.create_gmx_script(job)
        job_command = ['bash','remake_tpr.sh']
        subp = subprocess.Popen(job_command)
        subp.wait()
        job_command = 'gmx_mpi editconf -f start_conf.gro -o reference.pdb -n index.ndx<<<{}'.format(job.sp.filter_group[1])
        subp = subprocess.Popen(job_command, shell=True, executable='/bin/bash')
        subp.wait()

    copyfile('analysis_codes/calc_dist_matrix_for_frame.py',job.fn('calc_dist_matrix_for_frame.py'))
    copyfile('checkio.sh',job.fn('checkio.sh'))

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
            job_command = ['mpirun','-n','1','retriever','calc_dist_matrix_for_frame','analyze', str(n_frames)]
            generate_retriever = subprocess.Popen(job_command, stdout=retriever_out, stderr=retriever_out, shell=False)

        if 'plumed' in job.sp.job_type:
            job_command = ['mpirun','-n',str(job.sp.NPROCS),'gmx_mpi','mdrun','-v','-s','topol.tpr','-plumed','plumed.dat']
        else:
            job_command = ['mpirun','-n',str(job.sp.NPROCS),'gmx_mpi','mdrun','-v','-s','topol.tpr']
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
        job.document['total_time_s'] = t

        if 'plumed_ds_' in job.sp.job_type:
            generate_ds_server.kill()

        time.sleep(0)
        generate_iostats.kill()
    

@A4MDProject.operation
@A4MDProject.pre(processed)
@A4MDProject.post(post_processed)
@flow.directives(np=8)
def post_process(job):
    job.document['simulation_time_s'] = get_simulation_time_s(job)
    # analysis_time_s is recorded during analysis. So no need to do it here.
    job.document['data_management_time_s'] = get_data_management_time_s(job)
    #job.document['plumed_overhead_time_s'] = get_plumed_overhead_time_s(job)
    if job.sp.tau_profiling:
        with job:
            job_command = ['paraprof','--text','profile.tsv']
            generate_profile = subprocess.Popen(job_command, shell=False)
            generate_profile.wait()
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            profile_file = 'profile.tsv'
            df = pd.read_csv(profile_file,delimiter='\t',skipfooter=4)
            df['Function_short']=df.Function.apply(lambda x: x.split(' ')[0][:25])
            barplot(df,
                    'Exclusive TIME',
                    job.sp.data_dump_interval,
                    ylabel='Time [s]',
                    show_n_largest=10, 
                    multiplier=1e-6,
                    logscale=False)
            
if __name__ == '__main__':
    A4MDProject().main()
