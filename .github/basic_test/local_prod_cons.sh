#!/bin/bash

function usage {
  echo "Usage: flux start flux_prod_cons.sh <A4MD_INSTALL_PREFIX> <SPACK_VIEW> <HG_CONNECTION_STR>"
}

# Parse command line arguments

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

A4MD_INSTALL_PREFIX="$1"
SPACK_VIEW="$2"
HG_CONNECTION_STR="$3"

# Parameters for the run

# Number of writers
NWRITERS=2
# Number of readers per writer
NREADERS_PER_WRITER=1
# Number of writer processes
NP_WRITER=2
# Number of reader processes
NP_READER=2
# Number of server processes
NP_SERVER=1
# Number of steps
NSTEPS=10
# Window size
WINDOW=1
# Number of atoms
NATOMS=200
# Delay time
DELAY=0
# Number of iterations to wait for the server to startup
SERVER_TIMEOUT_ITERS=20

# Number of readers
NREADERS=$(( $NWRITERS*$NREADERS_PER_WRITER ))

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SPACK_VIEW/lib64:$SPACK_VIEW/lib:$A4MD_INSTALL_PREFIX/lib:$A4MD_INSTALL_PREFIX/lib64:$LD_LIBRARY_PATH
echo "Lib Path = $LD_LIBRARY_PATH"

# Set total number of DataSpaces clients
NCLIENTS=$(( $NREADERS*$NWRITERS ))

# Cleanup artifacts from previous runs
rm -rf __pycache__ conf dataspaces.conf log.* conf.ds

# Create DataSpaces server config
echo "## Config file for DSpaces Server
ndim = 1
dims = 1024
max_versions = 100
num_apps =" $NCLIENTS > dataspaces.conf

# Launch DataSpaces server
echo "-- Start DataSpaces server using $NP_SERVER processes"
server_cmd="mpirun -np $NP_SERVER $SPACK_VIEW/bin/dspaces_server $HG_CONNECTION_STR"
echo ${server_cmd}
${server_cmd} > server.out 2> server.err &
declare server_id=$!

# Wait briefly for the server to startup. If the server has not started by the timeout, error
sleep 1s
while [ ! -f conf.ds ]; do
  echo "-- conf.ds is not yet available from server. Sleep more"
  sleep 1s
done
# Wait for server to fully populate conf.ds
sleep 3s
# if [ ! -f "conf.ds" ]; then
#   echo "-- Server did not create conf.ds within $SERVER_TIMEOUT_ITERS iterations"
#   exit 2
# fi
echo "-- DataSpaces server initialized successfully"

example_dir="$A4MD_INSTALL_PREFIX/examples/sample_workflow"
client_id=0
group_id=0
# Run producers and consumers
echo "-- Start producers and consumers"
for (( i=1; i<=$NWRITERS; i++ )); do
  ((client_id=client_id+1))
  ((group_id=group_id+1))
  echo "-- Start producer $i"
  producer_cmd="mpirun -np $NP_WRITER ./producer dataspaces $client_id $group_id ./load.py extract_frame $NSTEPS $NATOMS $DELAY"
  echo ${producer_cmd}
  ${producer_cmd} > producer${i}.out 2> producer${i}.err &
  declare producer${i}_id=$!

  for (( j=1; j<=$NREADERS_PER_WRITER; j++ )); do
    ((client_id=client_id+1))
    echo "-- Start consumer application ${j} assocated with producer ${i}"
    consumer_cmd="mpirun -np $NP_READER ./consumer dataspaces $client_id $group_id ./compute.py analyze $NSTEPS"
    echo ${consumer_cmd}
    ${consumer_cmd} > consumer${i}_${j}.out 2> consumer${i}_${j}.err &
    declare consumer${i}_${j}_id=$!
  done
done

echo "-- Wait until all applications exit."

for (( i=1; i<=$NWRITERS; i++ )); do
  producer_id=producer${i}_id
  wait ${!producer_id} || exit 1
  echo "-- Producer $i exit."
done
echo "-- All producers exit."

for (( i=1; i<=$NWRITERS; i++ )); do
  for (( j=1; j<=$NREADERS_PER_WRITER; j++ )); do
    consumer_id=consumer${i}_${j}_id
    wait ${!consumer_id} || exit 1
    echo "-- Consumer ${i}_${j} exit."
  done
done
echo "-- All consumers exit."

echo "-- Kill DataSpaces server if it has not already shutdown"
kill -9 ${server_id}

echo "-- All applications exit."