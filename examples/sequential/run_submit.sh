
for i in 1 5000 10000 15000 20000
#for i in 20000
do 
  export dump_interval="$i"
  sbatch submit.sh
done
