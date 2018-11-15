for NPROCS in 1
do
  for L in 15
  do
    for i in 20000 15000 10000 5000 1000
    #for i in 1000
    do 
    for trial in 1 2 3 4 5
    #for trial in 1
      do
        export dump_interval="$i"
        export L="$L"
        export NPROCS="$NPROCS"
        export DATA_DIR=$PWD/'T_1_L_'$L'_dump_'$dump_interval'_trial_'$trial'_nprocs_'$NPROCS
        if [ ! -d $DATA_DIR ]; then
          sbatch submit.sh
        fi
      done
    done
  done
done
