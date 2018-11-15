import numpy as np
import os
from glob import iglob
import subprocess
import freud
from timeit import default_timer as timer
from scipy import stats
import json

an_times = []
def analyze(types, points, box_points, step):
    #print('-----======= Python : analyze ========-------')
    #print('box points',box_points);
    t=timer() 
    box = freud.box.Box(box_points[0],
        		box_points[1],
        		box_points[2],
        		box_points[3],
        		box_points[4],
        		box_points[5])
    voro = freud.voronoi.Voronoi(box, np.max(box.L)/2) 
    cells = voro.compute(box=box, positions=points).polytopes
    an_time = timer()-t
    an_times.append(an_time)
    if step>=20000:
        t=timer()
        print("In analyze: step: ",step)
        print('last frame position[0]',points[0])
        print('freud box', box)

        voro.computeVolumes()
        data = voro.volumes
        av = np.mean(data)
        ma = np.max(data)
        mi = np.min(data)
        print('mean {} min {} max {}'.format(av,mi,ma))
        np.savetxt('voro_volumes.txt',data)
        frq,edges = np.histogram(data,range=[0,50],bins=30)
        np.savetxt('voro_freq.txt',frq)
        np.savetxt('voro_edges.txt',edges)
        print('Number of voronoi cells',len(cells))
        an_write_time = timer()-t
        with open('signac_job_document.json', 'r') as f:
            job_document = json.load(f)

        job_document['analysis_time'] = np.sum(an_times)
        job_document['analysis_time_sem'] = stats.sem(an_times)
        job_document['analysis_output_time'] = np.sum(an_write_time)

        with open('signac_job_document.json', 'w') as f:
            f.write(json.dumps(job_document))

    return 0

if __name__ == "__main__":
    print('--------=========== Runnign analysis in python ===========-------------')

