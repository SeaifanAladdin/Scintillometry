module purge
module load gcc intel/15.0.2 openmpi/intel/1.6.4 python

for ((t = 1; t <=8; t++)); do
    mpirun -np 1 OMP_NUM_THREADS=$t python timeVaryThreads.py $n $m $p $num
done

