#!/bin/bash -l
#SBATCH --partition main 
#SBATCH --qos main-generic
#SBATCH -A parashar-003 
#SBATCH -N 3
#SBATCG -n 16
#SBATCH -J dataspaces
##SBATCH --gres=craynetwork:1

##SBATCH -C haswell 
##SBATCH -t 00:60:00
#module load python
##SBATCH --array=1-5


module purge
module load openmpi/2.1.3-gcc-8.1.0
module load python/3.6.3
#conda activate a4md
#export TAU_VERBOSE=1
export TAU_TRACK_SIGNALS=1
#export TAU_METRICS=TIME
#,PAPI_NATIVE_powercap:::ENERGY_UJ:ZONE0
#export PROFILEDIR=PROFILES
export THIS_DIR=`pwd`
echo $THIS_DIR

# Control will enter here if $DIRECTORY doesn't exist.
echo "DUMP_INTERVAL : $dump_interval"

if [ ! -d $DATA_DIR ]; then
  mkdir $DATA_DIR
  cp in.lj $DATA_DIR/
  cp calc_voronoi_for_frame.py $DATA_DIR/ 
  cp plumed.dat $DATA_DIR/ 
  cd $DATA_DIR
  START=$(date +%s.%N)
  srun -n 4 lmp_mpi -v T 1 -v d_int $dump_interval <in.lj 
  END=$(date +%s.%N)
  DIFF=$(echo "$END - $START" | bc)
  echo "SIM_TIME:" $DIFF
  sleep 5
  cd $THIS_DIR
fi 

