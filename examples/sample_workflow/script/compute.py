import numpy as np
from timeit import default_timer as timer
import math
import time

an_times = []

def analyze(types, xpoints,ypoints,zpoints, box_points, step):
    print('-----======= Python : analyze ({})========-------'.format(step))
    print(f'Compute::analyze:: writing retrieved cords to file')
    file_name='log.cords_from_retriever'
    with open(file_name,'a') as f:
         f.write(f'X:{xpoints}\n')
         f.write(f'Y:{ypoints}\n')
         f.write(f'Z:{zpoints}\n')
    t_start=timer() 
    points = np.vstack((xpoints,ypoints,zpoints)).T
    
    an_time = timer() - t_start
    print('analysis_only_time_per_step_ms : {}'.format(an_time*1000))
    an_times.append(an_time)

    return 0

if __name__ == "__main__":
    print('--------=========== Running analysis in python ===========-------------')
