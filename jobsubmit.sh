#!/bin/sh
# @ job_name           = 2048x50Padded
# @ job_type           = bluegene
# @ comment            = "n=2048, m=50, zero-padded"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 256
# @ wall_clock_limit   = 24:00:00
# @ bg_connectivity    = Torus
# @ queue 

source /scratch/s/scinet/nolta/venv-numpy/setup

NP=2048
OMP=8 ## Each core has 4 threads. Since RPN = 16, OMP = 4?
RPN=8

module purge

module load python/2.7.3
module load xlf/14.1 essl/5.1

cd /scratch/p/pen/seaifan/Scintillometry/src

echo "----------------------"
echo "STARTING in directory $PWD"
date
echo "np ${NP}, rpn ${RPN}, omp ${OMP}"

time OMP_NUM_THREADS=${OMP} runjob --np ${NP} --ranks-per-node=${RPN} --env-all : `which python` run_real.py yty2 0 140 2048 50 50 1

echo "ENDED"
date
