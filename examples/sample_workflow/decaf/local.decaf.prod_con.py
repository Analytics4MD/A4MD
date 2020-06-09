# a small 2-node example, just a producer and consumer

# --- include the following 4 lines each time ---

import networkx as nx
import os
import imp
import sys
wf = imp.load_source('workflow', os.environ['DECAF_PREFIX'] + '/python/decaf.py')

# --- set your options here ---

# path to .so module for dataflow callback functions
#mod_path = os.environ['DECAF_PREFIX'] + '/examples/direct/mpmd/mod_dflow.so'

# define workflow graph
# 2-node workflow
#
#    prod (4 procs) -> con (2 procs)
#
#  entire workflow takes 8 procs (2 dataflow procs between producer and consumer)
#  dataflow can be overlapped, but currently all disjoint procs (simplest case)

# --- Graph definition ---
nproducers = int(sys.argv[1])
nconsumers_per_producer = int(sys.argv[2])
job_name = "local.decaf.prod_con"

process_id = -1
for prod_id in range(nproducers):
	process_id += 1
	prod = wf.Node("producer"+str(prod_id), start_proc=process_id, nprocs=1, func='', cmdline='./producer decaf ' + job_name + '.json ./load.py extract_frame 5 200 0')
	outPort = prod.addOutputPort("out")

	for con_id in range(nconsumers_per_producer):
		process_id += 1
		con = wf.Node("consumer"+str(prod_id)+"."+str(con_id), start_proc=process_id, nprocs=1, func='', cmdline='./consumer decaf ' + job_name + '.json ./compute.py analyze 5')
		inPort = con.addInputPort("in")

		prod_con = wf.Edge(prod.getOutputPort("out"), con.getInputPort("in"), start_proc=0, nprocs=0, func='', prod_dflow_redist='count', dflow_con_redist='count')

# --- convert the nx graph into a workflow data structure and run the workflow ---

wf.processGraph(job_name)
