#!/bin/bash

## -----------=========== PARAMETERS =========-------------

## A4MD installation directory
A4MD=$SCRATCH/application/a4md/a4md/bin
## Number of ingesters
NWRITERS=2
## Ratio
NREADERS_PER_WRITER=1
## Number of consumers
NREADERS=$(( $NWRITERS*$NREADERS_PER_WRITER ))
## Number of producer processes
NP_WRITER=1
## Number of consumer processes
NP_READER=1
## Lock type
LOCK=2
## Number of DataSpaces servers
NSERVERS=1
## Number of steps
NSTEPS=10
## Number of atoms
NATOMS=200
## Delay time
DELAY=0

## -----------================================------------


## Number of ingesters and comsumers in total
NCLIENTS=$(( $NREADERS+$NWRITERS ))

## Clean up
rm -rf __pycache__ conf dataspaces.conf log.*

## Create dataspaces configuration file
echo "## Config file for DataSpaces
ndim = 1
dims = 1024
max_versions = 1
max_readers =" $NREADERS_PER_WRITER "
lock_type =" $LOCK > dataspaces.conf


## Run DataSpaces servers
echo "-- Start DataSpaces server on $NSERVERS PEs"
server_cmd="mpirun -np $NSERVERS $A4MD/dataspaces_server -s $NSERVERS -c $NCLIENTS"
echo ${server_cmd}
${server_cmd} &> log.server &
server_pid=$!

## Give some time for the servers to load and startup
sleep 1s
while [ ! -f conf ]; do
    echo "-- File conf is not yet available from server. Sleep more"
    sleep 1s
done
sleep 3s  # wait server to fill up the conf file
## Export the main server config to the environment
while read line; do
    export set "${line}"
done < conf
echo "-- Dataspaces Servers initialize successfully"
echo "-- DataSpaces IDs: P2TNID = $P2TNID   P2TPID = $P2TPID"
echo "-- Staging Method: $STAGING_METHOD"


client_id=0
group_id=0
## Run producer application
echo "-- Start producer applications"
for (( i=1; i<=$NWRITERS; i++ ))
do
    ((client_id=client_id+1))
    ((group_id=group_id+1)) 
    echo "-- Start producer application id $i"
    producer_cmd="mpirun -np $NP_WRITER ./chain_producer $client_id $group_id ./load.py extract_frame $NSTEPS $NATOMS $DELAY"
    echo ${producer_cmd}
    eval ${producer_cmd} &> log.producer${i} &
    declare producer${i}_pid=$!

    ## Run consumer application
    echo "-- Start consumer applications"
    for (( j=1; j<=$NREADERS_PER_WRITER; j++ ))
    do
        ((client_id=client_id+1))
        echo "-- Start consumer application id ${j} with respect to producer application id ${i}"
        consumer_cmd="mpirun -np $NP_READER ./chain_consumer $client_id $group_id ./compute.py analyze $NSTEPS"
        echo ${consumer_cmd}
        eval ${consumer_cmd} &> log.consumer${i}_${j} &
        declare consumer${i}_${j}_pid=$!
    done
done

echo "-- Wait until all applications exit."

for (( i=1; i<=$NWRITERS; i++ ))
do
    producer_pid=producer${i}_pid
    wait ${!producer_pid}
    echo "---- Producer id $i exit."
done
echo "---- All producers exit."
	
for (( i=1; i<=$NWRITERS; i++ ))
do
    for (( j=1; j<=$NREADERS_PER_WRITER; j++ ))
    do
        consumer_pid=consumer${i}_${j}_pid
        wait ${!consumer_pid}
        echo "---- Consumer id ${i}_${j} exit."
    done
done
echo "---- All consumers exit."

echo "-- Kill Dataspaces server externally."
kill -9 ${server_pid}

echo "-- All applications exit."
