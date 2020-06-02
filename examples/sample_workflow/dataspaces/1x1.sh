#!/bin/bash

## -----------=========== PARAMETERS =========-------------

## A4MD installation directory
A4MD=$SCRATCH/application/a4md/a4md/bin
## Number of jobs
NJOBS=1
## Number of consumers per ingester
NREADERS=1
## Number of ingesters
NWRITERS=1
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
NCLIENTS=$(( $NJOBS*($NREADERS+$NWRITERS) ))

## Clean up
rm -rf __pycache__ conf dataspaces.conf log.*

## Create dataspaces configuration file
echo "## Config file for DataSpaces
ndim = 1
dims = 1024
max_versions = 1
max_readers =" $NREADERS "
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

## Run producer application
echo "-- Start producer application"
producer_cmd="mpirun -np $NWRITERS ./producer dataspaces 1 1 ./load.py extract_frame $NSTEPS $NATOMS $DELAY"
echo ${producer_cmd}
eval ${producer_cmd} &> log.producer &
producer_pid=$!

## Run consumer application
echo "-- Start consumer application"
consumer_cmd="mpirun -n $NREADERS ./consumer dataspaces 2 1 ./compute.py analyze $NSTEPS"
echo ${consumer_cmd}
eval ${consumer_cmd} &> log.consumer &
consumer_pid=$!

echo "-- Wait until all applications exit."
#wait

wait $producer_pid
echo "---- Producer exit."
	
wait $consumer_pid
echo "---- Consumer exit."

echo "-- Kill Dataspaces server externally."
kill -9 ${server_pid}

echo "-- All applications exit."
