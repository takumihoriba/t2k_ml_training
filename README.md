# T2K ML Training

## Repo to train T2K data

### Set-up

Once you've cloned the directory, there is some setup to do.

You want to use anaconda to install some packages. Make a folder in your home directory (e.g. miniconda), navigate to it,  and run

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh 
```

This will download and install miniconda, with prompts to decide where to install. To load the conda environment used here, simply navigate to the top directory of the repo and run

```
conda env create --file=t2k_ml_train.yml
conda activate t2k_ml_train
```

This conda environment should give you access to most libraries needed in this repo. If running things locally, when in the main repo directory, one should run this every new shell, except when running WCSim:

```
conda activate t2k_ml_train
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

### Summarizing training

A few plots can be made, post-training, to see performance in training and test data. In order to run this, set

```
--doQuickPlots
--plotInput=[...]
--plotOutput=[...]
```

where plot input is the path to the directory where the logs from training were saved, and plot output is where you want to save the plots.

