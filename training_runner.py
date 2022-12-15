from WatChMaL.analysis.plot_utils import disp_learn_hist, disp_learn_hist_smoothed, compute_roc, plot_roc

import argparse
import debugpy
import h5py
import logging
import os
import csv
import numpy as np
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from WatChMaL.watchmal.engine.engine_classifier import ClassifierEngine
from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile
from analysis.utils.plotting import plot_legend
import analysis.utils.math as math
from runner_util import utils, train_config
from analysis.utils.binning import get_binning

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


def training_runner(rank, settings, kernel_size, stride):
    
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
    settings.initClassifier(kernel_size, stride)
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

def init_training(settings, kernel_size, stride):

    # Choose settings for utils class in util_config.ini
    if settings==0:
        print("Settings did not initialize properly, exiting...")
        exit

    os.environ['MASTER_ADDR'] = 'localhost'
    master_port = 12355
    # Automatically select port based on base gpu
    os.environ['MASTER_PORT'] = str(master_port)

    if settings.multiGPU:
        master_port += settings.gpuNumber[0]
        mp.spawn(training_runner, nprocs=len(settings.gpuNumber), args=(settings,kernel_size,))
    else:
        training_runner(0, settings, kernel_size, stride)

def efficiency_plots(settings, arch_name, newest_directory, plot_output):
    
    # retrieve test indices
    idx = np.array(sorted(np.load(str(newest_directory) + "/indices.npy")))
      
    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(settings.inputPath, "r")
    print(list(hy.keys()))
    angles = np.array(hy['angles'])[idx].squeeze()
    labels = np.array(hy['labels'])[idx].squeeze() 
    veto = np.array(hy['veto'])[idx].squeeze()
    energies = np.array(hy['energies'])[idx].squeeze()
    positions = np.array(hy['positions'])[idx].squeeze()
    directions = math.direction_from_angles(angles)

    # calculate number of hits 
    events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
    nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()

    # calculate additional parameters 
    towall = math.towall(positions, angles, tank_axis = 2)
    dwall = math.dwall(positions, tank_axis = 2)
    momentum = math.momentum_from_energy(energies, labels)

    # apply cuts, as of right now it should remove any events
    nhit_cut = nhits > 0 #25
    # veto_cut = (veto == 0)
    hy_electrons = (labels == 1)
    hy_muons = (labels == 2)
    basic_cuts = ((hy_electrons | hy_muons) & nhit_cut)

    # set class labels and decrease values within labels to match either 0 or 1 
    e_label = [0]
    mu_label = [1]
    labels = [x - 1 for x in labels]
        
    # get the bin indices and edges for parameters
    polar_binning = get_binning(np.cos(angles[:,0]), 10, -1, 1)
    az_binning = get_binning(angles[:,1]*180/np.pi, 10, -180, 180)
    mom_binning = get_binning(momentum, 10)
    dwall_binning = get_binning(dwall, 10)
    towall_binning = get_binning(towall, 10)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    #run = WatChMaLClassification(newest_directory, str(arch_name), labels, idx, basic_cuts, color="blue", linestyle='-')
    '''
    conv1x1 = '/fast_scratch/jsholdice/OutputPath/conv1x1'
    conv3x3 = '/fast_scratch/jsholdice/OutputPath/conv3x3'
    conv5x5 = '/fast_scratch/jsholdice/OutputPath/conv5x5'
    conv7x7 = '/fast_scratch/jsholdice/OutputPath/conv7x7'
    conv9x9 = '/fast_scratch/jsholdice/OutputPath/conv9x9'
    run_result = [WatChMaLClassification(conv1x1, 'Conv1x1', labels, idx, basic_cuts, color="blue", linestyle='-'),
                  WatChMaLClassification(conv3x3, 'Conv3x3', labels, idx, basic_cuts, color="red", linestyle='-'),
                  WatChMaLClassification(conv5x5, 'Conv5x5', labels, idx, basic_cuts, color="green", linestyle='-'),
                  WatChMaLClassification(conv7x7, 'Conv7x7', labels, idx, basic_cuts, color="black", linestyle='-'),
                  WatChMaLClassification(conv9x9, 'Conv9x9', labels, idx, basic_cuts, color="purple", linestyle='-'),]
    '''
    run = WatChMaLClassification(newest_directory, 'Conv1x1', labels, idx, basic_cuts, color="blue", linestyle='-')
    run_result = [run]
    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.999
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_fixed_efficiency(e_label, mu_label, muon_efficiency, select_labels = mu_label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False)
    e_az_fig, az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False)
    e_mom_fig, mom_ax = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False)
    e_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False)
    mu_az_fig, az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False)
    mu_mom_fig, mom_ax = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False)
    mu_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False)

    # save plots of effiency as a function of specific parameters
    e_polar_fig.savefig(plot_output + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(plot_output + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(plot_output + 'e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(plot_output + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(plot_output + 'e_towall_efficiency.png', format='png')
        
    mu_polar_fig.savefig(plot_output + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(plot_output + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(plot_output + 'mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(plot_output + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(plot_output + 'mu_towall_efficiency.png', format='png')

    return run


def main(kernel_size, stride):
    settings = utils()
    
    if args.doTraining:
        init_training(settings, kernel_size, stride) 
        
    if args.testParser:
        pass

    if args.doQuickPlots:
        
        _, arch_name = settings.getPlotInfo()
        newest_directory = max([os.path.join(args.plotInput,d) for d in os.listdir(args.plotInput)], key=os.path.getmtime)
        
        # create and save plots in specific training run file 
        plot_output = args.plotOutput + str(datetime.now()) + '/'
        os.mkdir(plot_output)

        # generate and save signal and background efficiency plots 
        #run = efficiency_plots(settings, arch_name, newest_directory, plot_output)
        run = efficiency_plots(settings, arch_name, newest_directory, plot_output)

        '''
        # Organizes data for csv file 
        input_header = ["INPUT HYPERPARAMETERS"]
        hyper_parameters = [['Network: '+str(settings.arch)], ['Train Batch Size: '+str(settings.TrainBatchSize)], ['Val Batch Size: '+str(settings.ValBatchSize)], 
                            ['Optimizer: '+ str(settings.optimizer)], ['Epochs: '+str(settings.epochs)], ['Restore Best State: '+str(settings.restoreBestState)],
                            ['Learning Rate: '+str(settings.lr)], ['Weight Decay: '+str(settings.weightDecay)]]
        output_header = ["OUTPUT RESULTS"]
        output_results = [['Avg Eval Accuracy: '+str(self.avg_eval_acc)], ['Avg Eval Loss: '+str(self.avg_eval_loss)]]

        # Writes csv file 
        with open (str(newest_directory) + 'Inputs_Outputs.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(input_header)
            writer.writerows(hyper_parameters)
            writer.writerow([])
            writer.writerow(output_header)
            writer.writerows(output_results)
        '''

        # plot training progression of training displaying the loss and accuracy throughout training and validation
        fig,ax1,ax2 = run.plot_training_progression()
        fig.tight_layout(pad=2.0) 
        fig.savefig(plot_output + 'log_test.png', format='png')

        # calculate softmax and plot ROC curve 
        softmax = np.load(newest_directory + '/softmax.npy')
        labels = np.load(newest_directory + '/labels.npy')
        fpr, tpr, thr = compute_roc(softmax, labels, 1, 0)
        plot_tuple = plot_roc(fpr,tpr,thr,'Electron', 'Muon', fig_list=[0,1,2], plot_label=arch_name)
        for i, plot in enumerate(plot_tuple):
            plot.savefig(plot_output + 'roc' + str(i) + '.png', format='png')

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    kernel_size = 1
    stride = 1
    print("MAIN")
    main(kernel_size, stride)


'''
elif args.doTraining:
    main()




elif args.testParser:
    settings = utils()

elif args.doQuickPlots:
    fig = disp_learn_hist(args.plotInput, losslim=2, show=False)
    fig.savefig(args.plotOutput+'resnet_test.png', format='png')
'''
