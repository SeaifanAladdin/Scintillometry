# Scintillometry

##Synopsis
The purpose of this project is to decompose a toeplitz matrix. Because this project will run on SciNet, a supercomputer, the program will need to have multi-threading and multi-core capabilities. 

##Example

###Extracting Data from your binned file

To extract your binned data, move to the src folder and use `extract_realData2.py`

The format is
```
$ python extract_realData2.py binnedDataFile numofrows numofcolms offsetn offsetm n m
```
where n is the number of blocks and m is the size of each block.

So for example. if I wanted numofrows= 2048, numofcols=330, offsetn= 0, offsetm = 140, n=4, m=8, I would call
```
python extract_realData2.py gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined 2048 330 0 140 4 8
```
This will create a folder at `./processedData/gate0_numblock_4_meff_32_offsetn_0_offsetm_140`

The name of the folder it's create is usually

`gate0_numblock_(n)_meff_(mx4)_offsetn_(offsetn)_offsetm_(offsetm)`

Inside this folder, there will be a `gate0_numblock_(n)_meff_(mx4)_offsetn_(offsetn)_offsetm_(offsetm)_toep.npy` file

There will also be n npy files. They will each represent a block of the toepletz matrix. The name of the file represent which block they represent. (so `0.npy` is the first block of the toepletz matrix with size 4mx4m)

Please note that even though *m* increases by a factor of 4, arguments will still take the *m* you specefied.

Finally, there is a checkpoint folder which I'll discuss later on.


###Applying the toepletz factorizor

### Locally

Now that we have our extracted the data we want, it's time to apply the toepletz factorizor.

First, let's do this on your local computer. Here, I'm using openmpi

```
$ module load mpi
$ module list
Currently Loaded Modulefiles:
  1) mpi/openmpi-x86_64
```
Then at `./src`, we want to use the python file `run_real.py`
```
mpirun --np num_of_processors python run_real.py method_name offsetn offsetm n m p pad
```
Note that due to scalability problems, num_of_processors must equal n. So this can be rewritten as 
```
mpirun --np n python run_real.py method_name offsetn offsetm n m p pad
```
method_name can be any of the following 5 methods: seq, wy1, wy2, yty1, yty2

offsetn and offsetm must equal what you had when using extract_realdata2.py

n and m are the number of blocks and size of each block respectively. They are also the same to what you used for extract_realdata2.py

p is the p parameter.

pad is whether there is padding involved or not. it can be 0 for false, or 1 for true.

So using the previous example, I would call

```
$ time mpirun --np 4 python run_real.py yty2 0 140 4 8 2 1
Loop 1
Loop 2
Loop 3
Loop 4
Loop 5
Loop 6
Loop 7

real	0m0.866s
user	0m1.149s
sys	0m1.040s
```

It prints the loop to let you know how far it progressed in the factoization. 

I also used the time command to time how long it takes to execute this.

What this returns is a file at `./results`, with the name

`gate0_numblock_(n)_meff_(mx4)_offsetn_(offsetn)_offsetm_(offsetm)_uc.npy`

So in our case, we have a file 

`gate0_numblock_4_meff_32_offsetn_0_offsetm_140_uc.npy`


###ON SciNet

Please refer to SciNet [BGQ wiki](https://wiki.scinet.utoronto.ca/wiki/index.php/BGQ) before continuing.


First, compress your processed data
```
$ cd processedData/
$ tar -zcvf processedData.tar.gz gate0_numblock_(n)_meff_(4xm)_offsetn_(offsetn)_offsetm_(offsetm)
```
Then move the compressed folder into scinet using your login information
```
$ scp processedData.tar.gz (scinetUsername)@login.scinet.utoronto.ca:~/
```

Now ssh to SciNet
```
$ ssh (scinetUsername)@login.scinet.utoronto.ca
```

Now move the compressed folder to bgqdev and ssh

```
$ scp processedData.tar.gz bgqdev:~/
$ ssh bgqdev
```

Now clone this github repository

```
$ git clone https://github.com/SeaifanAladdin/Scintillometry.git
```

Move the compressed folder into `Scintillometry/src/processedData` 

```
$ mv processedData.tar.gz Scintillometry/src/processedData/
$ cd Scintillometry/src/processedData/
$ tar -zxvf processedData.tar.gz
```

Now copy the scripts to `$SCRATCH` and cd there. `$SCRATCH` is for computation, but is not backed up so don't forget to copy your results back to `$HOME`

```
cp -r ~/Scintillometry/ $SCRATCH
cd $SCRATCH
```

###Debug mode (small jobs)

Request a block and import necessary modules
```
$ debugjob
```

Navigate to `~/Scintillometry/src` and import necessary modules

```
$ cd $SCRATCH/Scintillometry/src/
$ source /scratch/s/scinet/nolta/venv-numpy/setup ##This is needed for threading     
$ module purge
$ module load python/2.7.3
$ module load xlf/14.1 essl/5.1
```

Now run the script

```
time OMP_NUM_THREADS=$OMP runjob --np $NP --ranks-per-node=$RPN --env-all : `which python` run_real.py method_name offsetn offsetm n m p pad
```
`$NP` is the number of processes you want. Usually, `NP = n`. But if you have zero padding, you can choose to do `NP = 2n`

`$RPN` is the number of MPI processes per node

`RPM =1 , 2 , 4 , 8 , 16 , 32 , 64 ` and `ranks-per-node ≤ np`

`$OMP` is the number of threads per node. `(RPM * OMP) ≤ 64 `

As an example, you can run

```
time OMP_NUM_THREADS=16 runjob --np 8 --ranks-per-node=4 --env-all : `which python` run_real.py yty2 0 140 4 8 8 1
```

###Submitting Job (big jobs)

Will do later


###Plotting our results

We can now plot our results using code Niliou has written

For example, we could use 

```
$ python plot_real_basic.py resultName
```

or

```
$ python plot_real.py resultName
```


So in our example, we could plot using

```
$ python plot_real_basic.py gate0_numblock_4_meff_32_offsetn_0_offsetm_140
$ python plot_real.py gate0_numblock_4_meff_32_offsetn_0_offsetm_140 
```

