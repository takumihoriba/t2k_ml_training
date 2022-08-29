# t2k_ml_training

singularity image

singularity pull docker://triumfmlutils/baseml:v2.0.6

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

The file _t2k\_ml\_runner.py_ is the steering script for everything that can be done using this repo. It parses _args\_ml.txt_ for arguments on what to run and what directories to look into for input/output data. Possible arguments will be outlined in their respective sections.