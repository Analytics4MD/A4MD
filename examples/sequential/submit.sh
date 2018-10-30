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
#SBATCH --array=1-5

module purge
module load openmpi/2.1.3-gcc-8.1.0
conda activate a4md
#export TAU_VERBOSE=1
export TAU_TRACK_SIGNALS=1
#export TAU_METRICS=TIME
#,PAPI_NATIVE_powercap:::ENERGY_UJ:ZONE0
#export PROFILEDIR=PROFILES
export THIS_DIR=`pwd`
echo $THIS_DIR

# Control will enter here if $DIRECTORY doesn't exist.
echo "dump interval is $dump_interval"
export DATA_DIR=$PWD/'T_1_N_2048_dump_'$dump_interval'_trial_'$SLURM_ARRAY_TASK_ID
#

if [ ! -d $DATA_DIR ]; then
  mkdir $DATA_DIR
  cp in.lj $DATA_DIR/
  cp calc_voronoi_from_trajectory.py $DATA_DIR/ 
  cd $DATA_DIR
  START=$(date +%s.%N)
  mpirun -n 4 lmp_mpi -v T 1 -v d_int $dump_interval <in.lj 
  python calc_voronoi_from_trajectory.py
  END=$(date +%s.%N)
  DIFF=$(echo "$END - $START" | bc)
  echo "SIM_TIME:" $DIFF
  sleep 5
  cd $THIS_DIR
fi 

