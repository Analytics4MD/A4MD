import numpy as np
import os
from glob import iglob
from collections import defaultdict
from scipy import stats

def timestamp_to_seconds(t):
    return sum(float(n) * m for n, m in zip(reversed(t), (1, 60, 3600)))

def get_dump_interval(log_file):
    dump_interval=None
    with open(log_file) as fp:
            line = fp.readline()
            #print(line)
            while line:
                if 'DATA_DIR' in line:
                    values = line.split(':')
                    DATA_DIR_FULL = values[1]
                    DATA_DIR = values[1].split('/')[-1]
                    dump_interval = int(DATA_DIR.split('_')[5])
                    break
                    
                line = fp.readline()
    return dump_interval

def get_L(log_file):
    L=None
    with open(log_file) as fp:
            line = fp.readline()
            #print(line)
            while line:
                if 'DATA_DIR' in line:
                    values = line.split(':')
                    DATA_DIR_FULL = values[1]
                    DATA_DIR = values[1].split('/')[-1]
                    L = int(DATA_DIR.split('_')[3])
                    break
                    
                line = fp.readline()
    return L

def get_N(log_file):
    N=None
    with open(log_file) as fp:
            line = fp.readline()
            #print(line)
            while line:
                if 'Created ' in line:
                    values = line.split(' ')
                    if values[2] == 'atoms\n':
                        N = int(values[1])
                        break
                    
                line = fp.readline()
    return N

def get_total_time(log_file):
    total_time=None
    with open(log_file) as fp:
            line = fp.readline()
            #print(line)
            while line:
                if 'ETE_TIME' in line:
                    #print(line)
                    values = line.split(':')
                    total_time = float(values[1])
                    break
                    
                line = fp.readline()
    return total_time

def get_analysis_time(log_file, legend):
    analysis_time=0
    with open(log_file) as fp:
            line = fp.readline()
            while line:
                if legend == 'Plumed (Sequential)':
                    get_analysis_time_from_lammps = False
                    if get_analysis_time_from_lammps:
                        analysis_time_labels = ['Modify','Ouput']#'Pair','Neigh']#,'Comm','Other']#,'Modify','Ouput']
                        values = line.split('|')
                        if len(values)== 6 and any(label in values[0] for label in analysis_time_labels):
                                values = line.split('|')
                                analysis_time += float(values[2]) # Adding up times for all the analysis_time_labels
                    else:
                        if 'DISPATCH_UPDATE_TIME_rank_0' in line:
                            values = line.split(':')
                            #print(line)
                            analysis_time += float(values[1])/1e3 # convert millisecond to second
                        
                elif legend == 'Traditional':
                    if 'ANALYZE_TIME' in line:
                        #print(N,line)
                        values = line.split(':')
                        analysis_time = float(values[1])
                    
                line = fp.readline()
    return analysis_time

def get_data_dir(log_file):
    LOG_FILE_DIR=os.path.dirname(log_file)
    with open(log_file) as fp:
            line = fp.readline()
            #print(line)
            while line:
                if 'DATA_DIR' in line:
                    values = line.split(':')
                    DATA_DIR_FULL = values[1]
                    DATA_DIR = values[1].split('/')[-1]
                    break
                    
                line = fp.readline()
    return os.path.join(LOG_FILE_DIR,DATA_DIR[:-1])

def get_sim_time(log_file,legend): # Modify and Output is left out):
    sim_time=0
    DATA_DIR=None
    LOG_FILE_DIR=os.path.dirname(log_file)
    if legend=='Traditional':
        sim_time_labels = ['Pair','Neigh','Comm','Other','Modify','Output']
    elif legend == 'Plumed (Sequential)':
        sim_time_labels = ['Pair','Neigh','Comm','Other','Output']
        
    with open(log_file) as fp:
            line = fp.readline()
            #print(line)
            while line:
                if 'DATA_DIR' in line:
                    values = line.split(':')
                    DATA_DIR_FULL = values[1]
                    DATA_DIR = values[1].split('/')[-1]
                    break
                    
                line = fp.readline()
    prod_log = os.path.join(LOG_FILE_DIR,DATA_DIR[:-1],'log.prod')
    #print('Looking for log file in',prod_log)
    with open(prod_log) as fp:
            line = fp.readline()
            #print(line)
            while line:
                values = line.split('|')
                if len(values)== 6 and any(label in values[0] for label in sim_time_labels):
                    values = line.split('|')
                    sim_time += float(values[2])
                    
                line = fp.readline()
                
    return sim_time
