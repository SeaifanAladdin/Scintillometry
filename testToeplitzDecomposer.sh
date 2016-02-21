#!/bin/sh
# @ job_name           = FOLD_JB
# @ job_type           = bluegene
# @ comment            = "by-channel JB"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 64
# @ wall_clock_limit   = 24:00:00
# @ bg_connectivity    = Torus
# @ queue
# Launch all BGQ jobs using runjob   
#PBS -l nodes=10:ppn=8,walltime=0:40:00
#PBS -N np80_nodes4_ppn8

# load modules (must match modules used for compilation)
module purge
#module unload mpich2/xl python
#module load   python/2.7.3         binutils/2.23      bgqgcc/4.8.1       mpich2/gcc-4.8.1 fftw/3.3.3-gcc4.8.1 
#module load vacpp
module load bgqgcc/4.8.1 vacpp
module load mpich2/gcc-4.8.1 lapack 

module load python/2.7.3
module load xlf/14.1 essl/5.1

#module load hdf5/189-v18-mpich2-xlc
#module load bgqgcc/4.8.1 mpich2/gcc-4.8.1 python/2.7.3 

# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd ./src

# PIN THE MPI DOMAINS ACCORDING TO OMP
export I_MPI_PIN_DOMAIN=omp


#python interface.py
runjob --np 1 --ranks-per-node=1 --envs HOME=$HOME LD_LIBRARY_PATH=$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/ : /scinet/bgq/tools/Python/python2.7.3-20131205/bin/python2.7 Test/ToeplitzFactorizorTest.py

echo "ENDED"
date

