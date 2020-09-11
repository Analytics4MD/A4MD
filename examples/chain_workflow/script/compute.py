import numpy as np
from timeit import default_timer as timer
import math
import time

an_times = []

def analyze(types, xpoints,ypoints,zpoints, box_points, step):
    print('-----======= Python : analyze ({})========-------'.format(step))
    t_start=timer() 
    points = np.vstack((xpoints,ypoints,zpoints)).T
    
    an_time = timer() - t_start
    print('analysis_only_time_per_step_ms : {}'.format(an_time*1000))
    an_times.append(an_time)

    return 0

if __name__ == "__main__":
    print('--------=========== Running analysis in python ===========-------------')