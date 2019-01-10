import numpy as np
import os
from glob import iglob
import subprocess
import freud
from timeit import default_timer as timer
from scipy import stats
import json
import time

an_times = []
an_write_times = []
ddi = None
nsteps = None
with open('signac_statepoint.json', 'r') as f:
    sp = json.load(f)
    ddi = sp['data_dump_interval']
    nsteps = sp['simulation_time']

def analyze(types, xpoints,ypoints,zpoints, box_points, step):
    print('-----======= Python : analyze ({})========-------'.format(step))
    #print('box points',box_points);
    #print("");
    t=timer() 
    
    points = np.vstack((xpoints,ypoints,zpoints)).T
    box = freud.box.Box(box_points[0],
        		box_points[1],
        		box_points[2],
        		box_points[3],
        		box_points[4],
        		box_points[5])
    voro = freud.voronoi.Voronoi(box, np.max(box.L)/2) 
    cells = voro.compute(box=box, positions=points).polytopes

    voro.computeVolumes()
    data = voro.volumes
    av = np.mean(data)
    ma = np.max(data)
    mi = np.min(data)

    frq,edges = np.histogram(data,range=[0,50],bins=30)
    an_time = timer()-t
    an_times.append(an_time)

    t=timer()
    np.savetxt('voro_volumes.txt',data)
    np.savetxt('voro_freq.txt',frq)
    np.savetxt('voro_edges.txt',edges)
    an_write_time = timer()-t
    an_write_times.append(an_write_time)    
    if step>=nsteps:
        print('------============ reached end of analysis ({}) ==========------------'.format(step))
        with open('signac_job_document.json', 'r') as f:
            job_document = json.load(f)

        job_document['analysis_time_s'] = np.sum(an_times)
        job_document['analysis_time_s_sem'] = stats.sem(an_times)
        job_document['analysis_output_time_s'] = np.sum(an_write_times)
        job_document['analysis_output_time_s_sem'] = stats.sem(an_write_times)

        with open('signac_job_document.json', 'w') as f:
            f.write(json.dumps(job_document))

    return 0

if __name__ == "__main__":
    print('--------=========== Runnign analysis in python ===========-------------')

