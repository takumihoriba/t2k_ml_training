from WatChMaL.analysis.plot_utils import disp_learn_hist, disp_learn_hist_smoothed, compute_roc, plot_roc

import argparse
import debugpy
import h5py
import logging
import os
import numpy as np
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from WatChMaL.watchmal.engine.engine_classifier import ClassifierEngine
from analysis.classification import WatChMaLClassification
from analysis.utils.plotting import plot_legend
from runner_util import utils, train_config

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--doTraining", help="run training", action="store_true")
parser.add_argument("--doQuickPlots", help="run training", action="store_true")
parser.add_argument("--testParser", help="run training", action="store_true")
parser.add_argument("--plotInput", help="run training")
parser.add_argument("--plotOutput", help="run training")
parser.add_argument("--training_input", help="where training files are")
parser.add_argument("--training_output_path", help="where to dump training output")
args = parser.parse_args(['--training_input','foo','@args_training.txt',
                            '--plotInput','foo','@args_training.txt',
                            '--plotOutput','foo','@args_training.txt',
                            '--training_output_path','foo','@args_training.txt'])

logger = logging.getLogger('train')


def training_runner(rank, settings):

    print(f"rank: {rank}")
    #gpu = settings.gpuNumber[rank]
    world_size=1
    settings.numGPU=1
    if settings.multiGPU:
        world_size = len(settings.gpuNumber)
        settings.numGPU = len(settings.gpuNumber)
        settings.gpuNumber = settings.gpuNumber[rank]

    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    #Initialize training configuration, classifier and dataset for engine training
    settings.set_output_directory()
    settings.initTrainConfig()
    settings.initClassifier()
    settings.initOptimizer()
    #If labels do not start at 0, saves offset so that they are changed during training
    settings.checkLabels()

    data_config, train_data_loader, val_data_loader, train_indices, test_indices, val_indices = settings.initDataset(rank)
    model = nn.parallel.DistributedDataParallel(settings.classification_engine, device_ids=[settings.gpuNumber])
    engine = ClassifierEngine(model, rank, settings.gpuNumber, settings.outputPath)

    engine.configure_data_loaders(data_config, train_data_loader, val_data_loader, settings.multiGPU, 0, train_indices, test_indices, val_indices, settings)
    engine.configure_optimizers(settings)
    settings.save_options(settings.outputPath, 'training_settings')
    engine.train(settings)

def init_training():

    #Choose settings for utils class in util_config.ini
    settings = utils()
    if settings==0:
        print("Settings did not initialize properly, exiting...")
        exit

    os.environ['MASTER_ADDR'] = 'localhost'

    master_port = 12355
        
    # Automatically select port based on base gpu
    os.environ['MASTER_PORT'] = str(master_port)

    if settings.multiGPU:
        master_port += settings.gpuNumber[0]

        mp.spawn(training_runner, nprocs=len(settings.gpuNumber), args=(settings,))
    else:
        training_runner(0, settings)
    
def main():

    if args.doTraining:
        init_training()
        
    if args.testParser:
        settings = utils()

    if args.doQuickPlots:
        settings = utils()
        _, arch_name = settings.getPlotInfo()
        run_directory = '/fast_scratch/jsholdice/OutputPath/'
        newest_directory = max([os.path.join(run_directory,d) for d in os.listdir(run_directory)], key=os.path.getmtime)
        run_name = 'Test'
        run_result = WatChMaLClassification(newest_directory,run_name)
        fig,ax1,ax2 = run_result.plot_training_progression()
        fig.tight_layout(pad=2.0) 

        plot_output = args.plotOutput + str(datetime.now()) + '/'
        os.mkdir(plot_output)
        fig.savefig(plot_output+'log_test.png', format='png')
        softmax = np.load(args.plotInput+'softmax.npy')
        labels = np.load(args.plotInput+'labels.npy')
        fpr, tpr, thr = compute_roc(softmax, labels, 1, 0)
        plot_tuple = plot_roc(fpr,tpr,thr,'Electron', 'Muon', fig_list=[0,1,2], plot_label=arch_name)
        for i, plot in enumerate(plot_tuple):
            plot.savefig(plot_output+'roc'+str(i)+'.png', format='png')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    print("MAIN")
    main()


'''
elif args.doTraining:
    main()




elif args.testParser:
    settings = utils()

elif args.doQuickPlots:
    fig = disp_learn_hist(args.plotInput, losslim=2, show=False)
    fig.savefig(args.plotOutput+'resnet_test.png', format='png')
'''
