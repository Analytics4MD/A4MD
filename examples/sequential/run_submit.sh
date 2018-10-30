
for i in 20000 15000 10000 5000 1
#for i in 20000
do 
  export dump_interval="$i"
  sbatch submit.sh
done
