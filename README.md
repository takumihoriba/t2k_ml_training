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

You can use a singularity container to use the python packages needed to run this code. Simply run this everytime you log in:

```
singularity shell --nv -B /fast_scratch -B /data -B /home /fast_scratch/triumfmlutils/containers/base_ml_recommended.simg
```

### Runner file

The file _training\_runner.py_ is the steering script for everything that can be done using this repo. It parses _args\_training.txt_ for arguments on what to run and what directories to look into for input/output data. Possible arguments will be outlined in their respective sections.

### Training

If you want to run training, in _args\_training.txt_, set 

```
--doTraining
```

where training_input is a path and name to text file with absolute path of the directory where the training data is. 

#### Training configuration file

Use _util\_config.ini_ to choose all the settings for training. This includes input file path, output model path, which architecture, how many epochs, etc. 
This will be managed by the _utils_ class in _runner\_util.py_. Right now ResNet and PointNet are supported, with their own independent set of settings. The _utils_ class object is typically called _settings_, and is used throughout the code to call up settings from _util\_config.ini_ when needed.

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

To then train on that rootfile with the indices files you crated, in _util\_config.ini_ set InputPath to the path+name of the rootfile, and IndicesFile to a comma separated list of path+name of the .npz indices files you created with this script. 

There is another important file to change the options. From the main directory navigate to _WatChMaL/config/t2k\_resnet\_train.yaml_ (for ResNet training). Here you can change more options.

To know which file to use to change options: _training\_runner.py_ has a function called _init\_training_ which reads in _util\_config.ini_ and overwrites the options listed in training\_runner.py in _t2k\_resnet\_train.yaml_. The rest of the options are defined in the .yaml file, regardless of if they are defined in _util\_config.ini_.

### After Training

Files will be written out to the _OutputPath_ directory you specify in _util\_config.ini_. Additionally, some stats are quickly calculated at the end of each training fold and saved in _training_stats.xml_ in the output path directory. 

Once all trainings are done (if doing multiple folds), you can run some scripts to calculate some overall training stats and output some plots. To do this, in _args\_training.txt_ set the options

```
--doComparison
--comparisonFolder=
```

where comparisonFolder is the _outputPath_ where your training files were output. It should contain _training_stats.xml_ and all the other output files from training. The script will print out overall stats from training and create a sub-directory called _plots_ in _comparisonFolder_ showing some efficiency plots.

If you would like to compare to FiTQun performance you need to add a fitqun_combine.hy file to this folder and then those results will be automatically added to the generated plots. 

### Evaluation

You can run the evaluation step on an already trained network with options

```
--doEvaluation
--evaluationInputDir=
--evaluationOutputDir=
```

The input directory has to have a trained network weights to load, in a .pth file. Output will then have all the numpy files output.

### Adding Regession

To run training and evaluation with regression (predicting the x, y, and z position of the neutrino event as well as the usual event classification) just add the following flag

```
--regression=True
```

Once complete, you will then want to run plotting.regression_analysis(DIRNAME) where DIRNAME is the location where your checkpoints are stored.

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


### Additional Resoureces

Ashley Ferreira (Offboarding Document and Slides)[https://drive.google.com/drive/folders/1pb7OnIS1pF8SUVSqqXvf60lkWlUFJZvH?usp=sharing]