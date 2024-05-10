# T2K ML Training

## Repo to train T2K data

To clone the repo

```
git clone https://github.com/felix-cormier/t2k_ml_training.git
```

Since we're using the _WatChMaL_ submodule, navigate to the repo directory and initialize the submodule by

```
cd t2k_ml/
git submodule update --init --recursive
```

Once you've cloned the repo and initialized the submodule, there is some setup to do.

### Set-up

Once you've cloned the directory, there is some setup to do.

#### On Triumf-ml1

You can use a singularity container to use the python packages needed to run this code. Simply run this everytime you log in:

```
singularity shell --nv -B /fast_scratch_2/ -B /fast_scratch -B /data -B /home /fast_scratch/triumfmlutils/containers/base_ml_recommended.simg
```

If there are any packages missing you should be able to pip install them in the singularity container

#### On cedar

To make python environments, I followed the [compute canada webite](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment).

Specifically:

To create a python virtual environment, run the following, where ENV is your environment name

```
module load StdEnv/2020
module load python/3.10.2
module load scipy-stack
module load gcc/9.3.0
module load root/6.20.04
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index -r requirements.txt
```

This will load pyton packages, make a new virtual environment ENV, and download all needed packages. It may take a few minutes.

Now every time you want to use the environment, simply do

```
module load StdEnv/2020
module load python/3.10.2
module load scipy-stack
module load gcc/9.3.0
module load root/6.20.04
source ENV/bin/activate
```



### Runner file

The file _training\_runner.py_ is the steering script for everything that can be done using this repo. It parses _args\_training.txt_ for arguments on what to run and what directories to look into for input/output data. Possible arguments will be outlined in their respective sections.

### Training

If you want to run training, in _args\_training.txt_, set 

```
--doTraining
```


#### Training configuration file

Use _config/util\_config.ini_ to choose some settings for training. This includes input file path, output model path, which architecture, etc. 
This will be managed by the _utils_ class in _runner\_util.py_. Right now ResNet and PointNet are supported, with their own independent set of settings. The _utils_ class object is typically called _settings_, and is used throughout the code to call up settings from _config/util\_config.ini_ when needed.

For more fine-grained control of training, there is also _WatChMaL/config/t2k\_resnet\_train.yaml_. Here you can change even more options. Be careful, as the options chosen in _config/util\_config.ini_ will overwrite those in the yaml file.

#### Training on Triumf-ml1

In _config/util\_config.ini_ you want to make sure _batchSystem_ is set to False. Then go as normal. I recommend using a screen session for training as it may take a long time.

#### Training on Cedar

We can make use of the batch system on cedar to run training on GPU machines.

In _config/util\_config.ini_ you want to make sure _batchSystem_ is set to True.
Then in _t2k\_ml\_training\_job.sh_, on the line which copies a file to the GPU machine, you want to make sure it is the right file that you specify in _config/util\_config.ini_ as _InputPath_.
Make sure in _training\_runner.py_ the file you use as inputPath in the _init\_training_ function has the same end (e.g. digi\_combine.hy) as the file you want to use.

Finally you are ready to send the job. In a directory in \project or \scratch space do

```
sbatch --time=72:00:00 --gres=gpu:v100l:4 --mem=0 --cpus-per-task=32 --account=rpp-blairt2k --job-name=sk_pos_elec /home/fcormier/t2k/ml/training/t2k_ml_training/t2k_ml_training_job.sh
```

You can change the job name to a more meaningful one for job tracking purposes. This command asks for a whole machine with 4 GPUs, if this is overkill check [this](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm) for more comamnds.


### Preparing new files for training

When you have a new combined e/mu/gamma (or any combo) rootfile you want to use for training, you need to run a script to divide the indices into train/test/validation sets.
In _args\_training.txt_, you have to set  

```
--doIndices
```

to run this script. In addition, if you want to run multiple folds (https://machinelearningmastery.com/k-fold-cross-validation/), you can set the numFolds variable

```
--numFolds=3
```

this for example would divide your input dataset into three folds, you can then independently train on all three and calculate results on independent test sets. If you set the folds to 1, you have to specify a train/test/val split in _training\_runner.py_ (or leave it at default 70/15/15). Fore more than 1 fold train/test/val split variable is not used as the test/val is taken using number of folds (e.g. 3 folds 33.3% of indices used for test/val) 

You should set the variables

```
--inputIndices=
--outputIndicesPath=
```

in _args\_training.txt_ to specify the input roofile to calculate the indices from, and the output directory where we will save all the numpy files with indices. The script should generate one .npz file per fold you specify.

To then train on that rootfile with the indices files you crated, in _config/util\_config.ini_ set InputPath to the path+name of the rootfile, and IndicesFile to a comma separated list of path+name of the .npz indices files you created with this script. 

There is another important file to change the options. From the main directory navigate to _WatChMaL/config/t2k\_resnet\_train.yaml_ (for ResNet training). Here you can change more options.

To know which file to use to change options: _training\_runner.py_ has a function called _init\_training_ which reads in _config/util\_config.ini_ and overwrites the options listed in training\_runner.py in _t2k\_resnet\_train.yaml_. The rest of the options are defined in the .yaml file, regardless of if they are defined in _config/util\_config.ini_.


### Evaluation

You can run the evaluation step on an already trained network with options

```
--doEvaluation
--evaluationInputDir=
--evaluationOutputDir=
```

The input directory has to have a trained network weights to load, in a .pth file. Output will then have all the numpy files output.

### Analysis

To analyze the results of a training.

In _args\_training.txt_, you have to set  

```
--doAnalysis
```

Options to decide what to run are in the config file _config/analysis\_config.ini_. The analysis expects that evaluation on the test set has been run on that particular training run, either at the end of training or through the running Evaluation as described above. If there are errors due to files missing, it may be because evaluation of the test set has not been run.



### (OBSOLETE) Summarizing training

A few plots can be made, post-training, to see performance in training and test data. In order to run this, set

```
--doQuickPlots
--plotInput=[...]
--plotOutput=[...]
```

where plot input is the path to the directory where the logs from training were saved, and plot output is where you want to save the plots.


### (OBSOLETE) Running on Compute Canada Clusters

Preferably run this code on narval.computecanada.ca by

```
ssh username@narval.computecanada.ca
```

For code editing and light work you can run on the login node. But to run these functions with large
amounts of data, you should run it in an interactive batch job. Compute Canada uses the slurm schedul
er. To run an interactive batch job which only uses CPU, do:


```
srun --mem-per-cpu=4G --nodes=1 --ntasks-per-node=4 --time=08:00:00 --pty bash -i
```

where _mem-per-cpu_ is the amount of memory you request per tasks, _nodes_ is the number of entire no
des you request (usually 1), _ntasks-per-node_ is the number of CPUs per node you request, and _time_
 is the maximum amount of time your job will take. You should customize these requests to whatever yo
u think you will need, but this is a good baseline. Once you run this command, your shell will now be
 running in on of the compute nodes, and you will be able to run cod ehtat uses numerous CPUs and lar
ger amounts of memory.

It is possible that large files will use more memory than allocated. If so, the server will kill the
process. You will have to exit the interactive job (Ctrl-D) and rerun the _srun_ command with higher
_mem-per-cpu_.

### GPU interactive jobs

To start a GPU interactive job, the _srun_ command is

```
srun --mem-per-cpu=4G --nodes=1 --gpus-per-node=1 --ntasks-per-node=4 --time=8:00:00 --pty bash -i
```

where the options are the same as before, but _gpus-per-node_ has been added; this particular one requests exactly one GPU.


