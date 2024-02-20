#!/bin/bash
#FLUX: -N 1
#FLUX: --exclusive
#FLUX: -q pdebug
#FLUX: --output=test.out
#FLUX: --error=test.err
#FLUX: --job-name=a4md_test

## -----------=========== PARAMETERS =========-------------

## Spack view
SPACK_VIEW=/g/g90/lumsden1/ws/a4md_refactor/a4md_dspaces_1.8/.spack-env/view
## A4MD installation directory
A4MD=/g/g90/lumsden1/ws/a4md_refactor/A4MD/_dspaces2_test/bin
## Number of ingesters
NWRITERS=2
## Ratio
NREADERS_PER_WRITER=1
## Number of consumers
NREADERS=$(( $NWRITERS*$NREADERS_PER_WRITER ))
## Number of producer processes
NP_WRITER=2
## Number of consumer processes
NP_READER=2
## Lock type
LOCK=2
## Number of DataSpaces servers
NSERVERS=1
## Number of steps
NSTEPS=10
WINDOW=1
## Number of atoms
NATOMS=200
## Delay time
DELAY=0

## -----------================================------------

export LD_LIBRARY_PATH=$SPACK_VIEW/lib64:$SPACK_VIEW/lib:/usr/tce/packages/python/python-3.10.8/lib:/usr/tce/packages/boost/boost-1.80.0-mvapich2-2.3.7-gcc-12.1.1/lib:$LD_LIBRARY_PATH

## Number of ingesters and comsumers in total
NCLIENTS=$(( $NREADERS+$NWRITERS ))

## Clean up
rm -rf __pycache__ conf dataspaces.conf log.*

## Create dataspaces configuration file
echo "## Config file for DataSpaces
ndim = 1
dims = 1024
max_versions = 100
num_apps =" $NCLIENTS > dataspaces.conf


## Run DataSpaces servers
echo "-- Start DataSpaces server on $NSERVERS PEs"
server_cmd="flux submit --flags=waitable -N 1 -n $NSERVERS --output=server.out --error=server.err $SPACK_VIEW/bin/dspaces_server na+sm"
echo ${server_cmd}
declare server_id=$(${server_cmd})

## Give some time for the servers to load and startup
sleep 1s
while [ ! -f conf.ds ]; do
    echo "-- File conf is not yet available from server. Sleep more"
    sleep 1s
done
sleep 3s  # wait server to fill up the conf file
## Export the main server config to the environment
# while read line; do
#     export set "${line}"
# done < conf
echo "-- Dataspaces Servers initialize successfully"
# echo "-- DataSpaces IDs: P2TNID = $P2TNID   P2TPID = $P2TPID"
# echo "-- Staging Method: $STAGING_METHOD"


client_id=0
group_id=0
## Run producer application
echo "-- Start producer applications"
for (( i=1; i<=$NWRITERS; i++ ))
do
    ((client_id=client_id+1))
    ((group_id=group_id+1)) 
    echo "-- Start producer application id $i"
    producer_cmd="flux submit --flags=waitable -N 1 -n $NP_WRITER --output=producer${i}.out --error=producer${i}.err ./producer dataspaces $client_id $group_id ./load.py extract_frame $NSTEPS $NATOMS $DELAY"
    echo ${producer_cmd}
    declare producer${i}_id=$(${producer_cmd})

    ## Run consumer application
    echo "-- Start consumer applications"
    for (( j=1; j<=$NREADERS_PER_WRITER; j++ ))
    do
        ((client_id=client_id+1))
        echo "-- Start consumer application id ${j} with respect to producer application id ${i}"
        consumer_cmd="flux submit --flags=waitable -N 1 -n $NP_READER --output=consumer${i}_${j}.out --error=consumer${i}_${j}.err ./consumer dataspaces $client_id $group_id ./compute.py analyze $NSTEPS $WINDOW"
        echo ${consumer_cmd}
        declare consumer${i}_${j}_id=$(${consumer_cmd})
    done
done

echo "-- Wait until all applications exit."

for (( i=1; i<=$NWRITERS; i++ ))
do
    producer_id=producer${i}_id
    flux job wait ${!producer_id}
    echo "---- Producer id $i exit."
done
echo "---- All producers exit."
	
for (( i=1; i<=$NWRITERS; i++ ))
do
    for (( j=1; j<=$NREADERS_PER_WRITER; j++ ))
    do
        consumer_id=consumer${i}_${j}_id
        flux job wait ${!consumer_id}
        echo "---- Consumer id ${i}_${j} exit."
    done
done
echo "---- All consumers exit."

echo "-- Kill Dataspaces server externally."
flux job kill -s 9 ${server_id}

echo "-- All applications exit."
