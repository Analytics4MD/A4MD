
for i in 20000 15000 10000 5000 1000 500 1
#for i in 20000
do 
  for trial in 1 2 3 4 5
  do
    export dump_interval="$i"
    export DATA_DIR=$PWD/'T_1_N_2048_dump_'$dump_interval'_trial_'$trial
    if [ ! -d $DATA_DIR ]; then
      sbatch submit.sh
    fi
  done
done
