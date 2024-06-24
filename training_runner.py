#from WatChMaL.analysis.plot_utils import disp_learn_hist, disp_learn_hist_smoothed, compute_roc, plot_roc

import argparse
#import debugpy
#import h5py
import logging
import os  
import csv
import numpy as np
from datetime import datetime
import itertools

import subprocess


#from analysis.classification import WatChMaLClassification
#from analysis.classification import plot_efficiency_profile
#from analysis.utils.plotting import plot_legend
import WatChMaL.analysis.utils.math as math

from analyze_output.analyze_regression import analyze_regression
from analyze_output.analyze_multiple_regression import MultiRegressionAnalysis, analyze_multiple_regression
from analyze_output.analyze_classification import analyze_classification

from runner_util import utils, analysisUtils, train_config, make_split_file
from WatChMaL.analysis.utils.binning import get_binning

import re


# from torchmetrics import AUROC, ROC

#from lxml import etree

import hydra

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--doTraining", help="run training", action="store_true")
parser.add_argument("--doFiTQun", help="run fitqun results", action="store_true")
parser.add_argument("--doEvaluation", help="run evaluation on already trained network", action="store_true")
parser.add_argument("--doMultiEvaluations", help="run evaluation on already trained network", action="store_true")
parser.add_argument("--doComparison", help="run comparison", action="store_true")
parser.add_argument("--doQuickPlots", help="Make performance plots", action="store_true")
parser.add_argument("--doAnalysis", help="run analysis of ml and/or fitqun", action="store_true")
parser.add_argument("--doMultiAnalyses", help="run analysis of ml and/or fitqun", action="store_true")
parser.add_argument("--doIndices", help="create train/val/test indices file", action="store_true")
parser.add_argument("--testParser", help="run training", action="store_true")
parser.add_argument("--plotInput", help="run training")
parser.add_argument("--comparisonFolder", help="run training")
parser.add_argument("--numFolds", help="run training")
parser.add_argument("--indicesInput", help="run training")
parser.add_argument("--evaluationInputDir", help="which training directory to get network for evaluation")
parser.add_argument("--evaluationOutputDir", help="where to dump evaluation results")
parser.add_argument("--indicesOutputPath", help="run training")
parser.add_argument("--plotOutput", help="run training")
parser.add_argument("--training_input", help="where training files are")
parser.add_argument("--training_output_path", help="where to dump training output")
args = parser.parse_args(['--training_input','foo','@args_training.txt',
                            '--plotInput','foo','@args_training.txt',
                            '--comparisonFolder','foo','@args_training.txt',
                            '--plotOutput','foo','@args_training.txt',
                            '--indicesInput','foo','@args_training.txt',
                            '--indicesOutputPath','foo','@args_training.txt',
                            '--evaluationInputDir','foo','@args_training.txt',
                            '--evaluationOutputDir','foo','@args_training.txt',
                            '--numFolds','foo','@args_training.txt',
                            '--training_output_path','foo','@args_training.txt'])
logger = logging.getLogger('train')



def training_runner(rank, settings, kernel_size, stride):

    import torch
    import torch.multiprocessing as mp
    import torch.nn as nn
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


def init_training():
    """Reads util_config.ini, constructs command to run 1 training
    """
    onCedar=False

    settings = utils()
    settings.set_output_directory()
    default_call = ["python", "WatChMaL/main.py", "--config-name="+settings.configName] 
    indicesFile = check_list_and_convert(settings.indicesFile)
    #Make sure the name of file matches the one you copy/set in util_config.ini
    if settings.batchSystem:
        inputPath = [os.getenv('SLURM_TMPDIR') + '/digi_combine.hy'] 
    else:
        inputPath = check_list_and_convert(settings.inputPath)
    featureExtractor = check_list_and_convert(settings.featureExtractor)
    lr = check_list_and_convert(settings.lr)
    lr_decay = check_list_and_convert(settings.lr_decay)
    weightDecay = check_list_and_convert(settings.weightDecay)
    stride = check_list_and_convert(settings.stride)
    kernelSize = check_list_and_convert(settings.kernel)
    perm_output_path = settings.outputPath
    variable_list = ['indicesFile', 'inputPath', 'learningRate', 'weightDecay', 'learningRateDecay', 'featureExtractor',  'stride', 'kernelSize']
    for x in itertools.product(indicesFile, inputPath, lr, weightDecay, lr_decay, featureExtractor, stride, kernelSize):
        default_call = ["python", "WatChMaL/main.py", "--config-name="+settings.configName] 
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y-%H%M%S")
        #dt_string = '20092023-101855'
        settings.outputPath = perm_output_path+'/'+dt_string+'/'
        print(f'TRAINING WITH\n input file: {x[1]} \n indices file: {x[0]}\n learning rate: {x[2]}\n learning rate decay: {x[4]}\n weight decay: {x[3]}\n feature extractor: {x[5]}\n output path: {settings.outputPath}')
        default_call.append("data.split_path="+x[0])
        default_call.append("data.dataset.h5file="+x[1])
        default_call.append("tasks.train.optimizers.lr="+str(x[2]))
        default_call.append("tasks.train.optimizers.weight_decay="+str(x[3]))
        default_call.append("tasks.train.scheduler.gamma="+str(x[4]))
        default_call.append("model._target_="+str(x[5]))
        default_call.append("model.stride="+str(x[6]))
        default_call.append("model.kernelSize="+str(x[7]))
        default_call.append("hydra.run.dir=" +str(settings.outputPath))
        default_call.append("dump_path=" +str(settings.outputPath))
        print(default_call)
        subprocess.call(default_call)
        end_training(settings, variable_list, x)


def check_list_and_convert(input):
    if type(input) is not list:
        output = [input]
    else:
        output = input
    return output

def end_training(settings, variable_list=[], variables=[]):

    softmaxes = np.load(settings.outputPath+'/'+'softmax.npy')
    labels = np.load(settings.outputPath+'/'+'labels.npy')
    print(f'Unique labels in test set: {np.unique(labels,return_counts=True)}')

    auroc = AUROC(task="binary")
    auc = auroc (torch.tensor(softmaxes[:,1]),torch.tensor(labels))
    print(f'AUC: {auc}')
    if len(np.unique(labels)) < 2:
        roc = ROC(task="binary")
        fpr, tpr, thresholds = roc(torch.tensor(softmaxes[:,1]), torch.tensor(labels))
        for i, eff in enumerate(tpr):
            #From SK data quality paper, table 13 https://t2k.org/docs/technotes/399/v2r1
            if eff > 0.99876:
                print(f'tpr: {eff}, bkg rej: {1/fpr[i]}')
                bkg_rej = 1/fpr[i]
                break
    else:
        roc = ROC(task="multiclass", num_classes = len(np.unique(labels)))
        fpr, tpr, thresholds = roc(torch.tensor(softmaxes), torch.tensor(labels))
    bkg_rej = 0

    root = etree.Element('Training')
    level1_stats = etree.SubElement(root, 'Stats')
    level2 = etree.SubElement(level1_stats, 'AUC', var=str(float(auc)))
    level2 = etree.SubElement(level1_stats, 'Bkg_Rejection', var=str(float(bkg_rej)))
    level1_var = etree.SubElement(root, 'Variables')
    for name, var in zip(variable_list, variables):
        level2 = etree.SubElement(level1_var, name, var=str(var))
    level1_files = etree.SubElement(root, 'Files')
    level2 = etree.SubElement(level1_files, 'inputPath', var=settings.inputPath)
    tree = etree.ElementTree(root)
    tree.write(settings.outputPath+'training_stats.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")
    



#if args.doComparison:
#    compare_outputs(args.comparisonFolder)

if args.doIndices:
    make_split_file(args.indicesInput, train_val_test_split=[0.05,0.05], output_path=args.indicesOutputPath, nfolds=args.numFolds, seed=0)

#settings = utils()
#kernel_size = settings.kernel
#stride = settings.stride

if args.doTraining:
    init_training() 

if args.doFiTQun:
    fitqun_regression_results()

if args.doEvaluation:
    settings = utils()
    settings.outputPath = args.evaluationOutputDir
    settings.set_output_directory()
    # default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval"] 
    # default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_dead"] 
    default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_classifier_dead"] 
    

    indicesFile = check_list_and_convert(settings.indicesFile)
    perm_output_path = settings.outputPath

    # default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_dead"] 


    settings.outputPath = args.evaluationInputDir
    default_call.append("hydra.run.dir=" +str(args.evaluationInputDir))
    default_call.append("dump_path=" +str(args.evaluationOutputDir))
    print(default_call)
    subprocess.call(default_call)
    #end_training(settings)

def multiAnalyses_helper(evalOutputDir=None, sort_by_percent=True):
    '''
    Helper function to retrieve directory names, seed values, iteration values, and percents for multi analyses.
    '''
    if evalOutputDir is None:
        evalOutputDir = args.evaluationOutputDir
    call = ['ls', evalOutputDir]
    res = subprocess.run(call ,capture_output=True, text=True)
    sd_names = re.findall(r'^(multiEval_seed.*)', res.stdout, re.MULTILINE)
    seeds = re.findall(r'^multiEval_seed_(\d*)_.*', res.stdout, re.MULTILINE)
    itrs = re.findall(r'^multiEval_seed_.*_(\d*)th_itr_.*', res.stdout, re.MULTILINE)
    percents = re.findall(r'^multiEval_seed_.*itr_(\d*)_percent_.*', res.stdout, re.MULTILINE)
    percents = [int(p) for p in percents]

    if sort_by_percent:
        comb = list(zip(sd_names, percents, seeds, itrs))
        sorted_comb = sorted(comb, key=lambda x:x[1]) # sorted based on percents
        sorted_sd_names = [x[0] for x in sorted_comb]
        sorted_seeds = [x[2] for x in sorted_comb]
        sorted_itrs = [x[3] for x in sorted_comb]
        sorted_percents = sorted(percents)

        return sorted_sd_names, sorted_seeds, sorted_itrs, sorted_percents
    else:
        return sd_names, seeds, itrs, percents

if args.testParser:
    settings = analysisUtils()
    settings.set_output_directory()
        
    sub_dir_names, _, _, percents = multiAnalyses_helper(args.evaluationOutputDir)

    # settings.doRegression = True
    # settings.mlPath = '/data/thoriba/t2k/eval/oct20_eMuPosPion_0dwallCut_flat_1/09052024-171021_regress/22032024-142255/'
    # settings.outputPlotPath='/data/thoriba/t2k/plots/plots_regression_test2/'

    if settings.doRegression:
        # because we always compute plots, each analysis takes about 30 sec to 1 min.

        print('regress')

        # sub_dir_names = [
        #     # 'multiEval_seed_0_0th_itr_0_percent_20240607172604',
        #     'multiEval_seed_0_0th_itr_10_percent_20240608021345',
        #     'multiEval_seed_0_0th_itr_3_percent_20240607174304',
        #     # 'multiEval_seed_0_0th_itr_5_percent_20240607215813',
        #     'multiEval_seed_10_10th_itr_10_percent_20240608050420',
        #     # 'multiEval_seed_10_10th_itr_3_percent_20240607203305',
        #     # 'multiEval_seed_10_10th_itr_5_percent_20240608004832'
        # ]

        # percents = [10, 3, 10]
        
        # analyze_multiple_regression(settings, sub_dir_names, percents)
        # mra = MultiRegressionAnalysis(settings=settings, sub_dir_names=sub_dir_names, percents=percents)
        mra = MultiRegressionAnalysis(settings=settings, sub_dir_names=sub_dir_names[:1], percents=percents[:1])
        
        # mra.plot_errorbars(file_path=settings.outputPlotPath + 'reg_analysis_metrics.csv')
        # mra.compute_bias_summary_stats()
        mra.plot_resdiual_scatter('energy', 'Longitudinal')
        # mra.plot_resdiual_scatter('visible energy', 'Longitudinal')
        # mra.plot_resdiual_scatter('nhit', 'Longitudinal')
        # mra.plot_resdiual_scatter('towall', 'Longitudinal')
        # mra.plot_resdiual_scatter('total_charge', 'Longitudinal')


        # mra = MultiRegressionAnalysis(settings=settings, sub_dir_names=sub_dir_names, percents=percents)

        # mra.plot_errorbars(file_path='')


    if settings.doClassification:
        pass

if args.doAnalysis:
    settings = analysisUtils()
    settings.set_output_directory()

    if settings.doRegression:
        #analyze_regression(settings.inputPath, settings.mlPath, settings.fitqunPath, settings.particleLabel, settings.target, settings.outputPlotPath)
        analyze_regression(settings)
        # analyze_regression_no_plots(settings)

    if settings.doClassification:
        analyze_classification(settings)

if args.doMultiAnalyses:
    settings = analysisUtils()
    settings.set_output_directory()
    
    sub_dir_names, _, _, percents = multiAnalyses_helper(args.evaluationOutputDir)

    if settings.doRegression:
        analyze_multiple_regression(settings, sub_dir_names[0], percents[0])

    if settings.doClassification:
        pass




if args.doQuickPlots:
    
    _, arch_name = settings.getPlotInfo()
    newest_directory = max([os.path.join(args.plotInput,d) for d in os.listdir(args.plotInput)], key=os.path.getmtime)
    
    # create and save plots in specific training run file 
    plot_output = args.plotOutput + str(datetime.now()) + '/'
    os.mkdir(plot_output)

    # generate and save signal and background efficiency plots 
    #run = efficiency_plots(settings, arch_name, newest_directory, plot_output)
    efficiency_plots(settings, arch_name, newest_directory, plot_output)
    
    
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
    '''

def run_evaluation(count=1, dead_pmt_seed=5, dead_pmt_rate=.03, config_name='t2k_resnet_eval_classifier_dead'):
    '''
    Runs evaluation process for one time with parameter values, specific to 'turning off PMTs'
    Returns command line output of main.py in WatChMal.
    Default is for classification.
    '''
    
    print(f"run_evaluation(count={count}, dead_pmt_seed={dead_pmt_seed}, dead_pmt_rate={dead_pmt_rate})")


    settings = utils()
    settings.outputPath = args.evaluationOutputDir
    settings.set_output_directory()
    default_call = ["python", "WatChMaL/main.py", f"--config-name={config_name}"] 
    indicesFile = check_list_and_convert(settings.indicesFile)
    perm_output_path = settings.outputPath

    # default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_classifier"] 
    settings.outputPath = args.evaluationInputDir
    

    default_call.append("hydra.run.dir=" +str(args.evaluationInputDir))
    default_call.append("dump_path=" +str(args.evaluationOutputDir))
    print(default_call)
    default_call += [f"data.dataset.dead_pmt_rate={dead_pmt_rate if dead_pmt_rate > 0. else 'null'}", f"data.dataset.dead_pmt_seed={dead_pmt_seed}"]

    result = subprocess.run(default_call, capture_output=True, text=True)
    print(f'{default_call} was called. No problem. ')
    if result.stderr is not None:
        print("Error: ", result.stderr)
    print("Output: ", result.stdout)
    return result

def copy_npy(i, s, p, mode='classification'):
    '''
    Copies *.npy files and .hydra directory (and its files inside) into a new folder
    Returns nothing.

    Parameters
    ----------
    i: iteration
    s: seed
    p: probability
        In [0, 1]
    mode: either classification or regression
    '''
    if mode is None:
        print('mode must not be null')
        raise ValueError()

    date_time_str = datetime.today().strftime("%Y%m%d%H%M%S")
    dir_name = args.evaluationOutputDir + f'multiEval_seed_{s}_{i}th_itr_{round(p*100)}_percent_{date_time_str}/'
    os.makedirs(dir_name, exist_ok=True)
    source_dir = args.evaluationOutputDir
    
    # copy result files of evaluation
    cp_command = []
    if mode == 'classification' or mode[0] == 'c':
        cp_command = ["cp", f"{source_dir}/indices.npy",f"{source_dir}/labels.npy",f"{source_dir}/softmax.npy", dir_name]
    elif mode == 'regression' or mode[0] == 'r':
        cp_command = ["cp", f"{source_dir}/indices.npy",f"{source_dir}/labels.npy",f"{source_dir}/positions.npy",f"{source_dir}/predicted_positions.npy", dir_name]
    else:
        print('mode must be either classification or regression')
        raise ValueError()

    result = subprocess.run(cp_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # copy .hydra files recursively
    cp_command = ["cp", "-r", f"{source_dir}/.hydra", dir_name]
    result = subprocess.run(cp_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    return dir_name

# def copy_npy_regress(i, s, p):
#     '''
#     Copies *.npy files and .hydra directory (and its files inside) into a new folder
#     Returns nothing.

#     Parameters
#     ----------
#     i: iteration
#     s: seed
#     p: probability
#         In [0, 1]

#     '''
#     date_time_str = datetime.today().strftime("%Y%m%d%H%M%S")
#     dir_name = args.evaluationOutputDir + f'multiEval_seed_{s}_{i}th_itr_{round(p*100)}_percent_{date_time_str}/'
#     os.makedirs(dir_name, exist_ok=True)
#     source_dir = args.evaluationOutputDir
    
#     # copy result files of evaluation
#     cp_command = ["cp", f"{source_dir}/indices.npy",f"{source_dir}/labels.npy",f"{source_dir}/positions.npy",f"{source_dir}/predicted_positions.npy", dir_name]
    
#     result = subprocess.run(cp_command, capture_output=True, text=True)
#     print(result.stdout)
#     print(result.stderr)

#     # copy .hydra files recursively
#     cp_command = ["cp", "-r", f"{source_dir}/.hydra", dir_name]
#     result = subprocess.run(cp_command, capture_output=True, text=True)
#     print(result.stdout)
#     print(result.stderr)

#     return dir_name



def multi_evaluations_regression(settings, probs=[.03, .05, .1], itr=10, config_name='t2k_resnet_eval_dead', one_itr_for_zero_percent=True):
    losses = None
    avg_metrics = None
    
    for prob in probs:
        sums = None # np.zeros((1, 1)) # loss
        for i in range(itr):
            s = i # seed
            r1 = run_evaluation(i, s, prob, config_name)
            print(r1.stdout)
            print(r1.stderr)
            # accuracy and loss
            loss_curr = re.findall(r'Average evaluation .*: (\d*\.\d*)', r1.stdout)
            loss_curr = np.array([float(m) for m in loss_curr])

            if losses is None:
                losses = np.insert(loss_curr, 0, round(prob*100))
            else:
                losses = np.vstack((losses, np.insert(loss_curr, 0, round(prob*100))))
            
            if sums is None:
                sums = loss_curr
            else:
                sums += loss_curr

            dir_name = copy_npy(i, s, prob, 'regression')
            np.savetxt(dir_name + 'loss.csv', loss_curr, delimiter=',')

            if prob == 0. and one_itr_for_zero_percent:
                print('For 0%, only one iteration is computed')
                break
              
        print(f"[average loss] for dead prob of {prob} and {itr} iterations")
        print(sums/itr)
        if avg_metrics is None:
            avg_metrics = sums / itr
        else:
            avg_metrics = np.vstack((avg_metrics, sums/itr))
    
    print("avg metrics:", avg_metrics)
    # np.savetxt('avg_metrics.csv', avg_metrics, delimiter=',')
    date_time_str = datetime.today().strftime("%Y%m%d%H%M%S")
    np.savetxt(args.evaluationOutputDir + 'avg_metrics_' + date_time_str +'.csv', avg_metrics, delimiter=',')
    avg_metrics_percent = np.insert(avg_metrics, 0, np.array(probs), axis = 1)
    np.savetxt(args.evaluationOutputDir + 'avg_metrics_with_percent' + date_time_str +'.csv', avg_metrics_percent, delimiter=',')
    np.savetxt(args.evaluationOutputDir + 'loss_log_' + date_time_str +'.csv', losses, delimiter=',')

    return avg_metrics_percent


def multi_evaluations_classification(settings, probs=[.03, .05, .1], itr=10, config_name='t2k_resnet_eval_classifier_dead', one_itr_for_zero_percent=True):
    accuracy_summary = None
    loss_summary = None
    
    for prob in probs:
        accuracies_per_prob = []
        losses_per_prob = []
        for i in range(itr):
            # set seed and run evaluation. print output
            s = i
            r1 = run_evaluation(i, s, prob, config_name)
            print(r1.stdout)
            print(r1.stderr)

            # get loss and accuacy from the output
            acc_loss_curr = re.findall(r'Average evaluation .*: (\d*\.\d*)', r1.stdout)
            acc_loss_curr = np.array([float(m) for m in acc_loss_curr])

            # collect accuracy and loss for this iteration
            accuracies_per_prob.append(acc_loss_curr[1])
            losses_per_prob.append(acc_loss_curr[0])
            
            # copy evaluation ouputs to a new folder (sub-directory); save accuracy and loss as csv
            dir_name = copy_npy(i, s, prob, 'classification')
            np.savetxt(dir_name + 'accuracy_loss'+'.csv', acc_loss_curr, delimiter=',')

            if prob == 0. and one_itr_for_zero_percent:
                print('For 0%, only one iteration is computed')
                break

        acc_summary_percent  = summary_stats_1d(accuracies_per_prob)
        loss_summary_percent = summary_stats_1d(losses_per_prob)

        acc_summary_percent  = np.insert(acc_summary_percent,  0, round(prob * 100))
        loss_summary_percent = np.insert(loss_summary_percent, 0, round(prob * 100))
        
        
        accuracy_summary = np.vstack((accuracy_summary, acc_summary_percent)) if accuracy_summary is not None else acc_summary_percent 
        loss_summary = np.vstack((loss_summary, loss_summary_percent)) if loss_summary is not None else loss_summary_percent
        

        print("summary stats for accuracy: ", accuracy_summary)
        print("summary stats for loss: ", loss_summary)
    
    date_time_str = datetime.today().strftime("%Y%m%d%H%M%S")
    np.savetxt(args.evaluationOutputDir + 'accuracy_summary_stats_per_percent' + date_time_str +'.csv', accuracy_summary, delimiter=',')
    np.savetxt(args.evaluationOutputDir + 'loss_summary_stats_per_percent' + date_time_str +'.csv', loss_summary, delimiter=',')

    return accuracy_summary

def summary_stats_1d(l):
    '''
    takes a list and returns a (7,) numpy array containing a statistic in each column
    Min, 25th percentile (Q1), 50th percentile (Q2, median), Mean, 75th percentile (Q3), Max, SD

    Params: l: assume l is a list
    ''' 
    
    if not isinstance(l, list):
        raise TypeError(f"Expected a list, but got {type(l).__name__}")

    x = np.array(l) 
    
    return np.array([
        np.min(x),
        np.percentile(x, 25),
        np.percentile(x, 50),
        np.mean(x),
        np.percentile(x, 75),
        np.max(x),
        np.std(x)
    ])

if args.doMultiEvaluations:
    '''
    Modify probs and itr, and flag variables to customize your evaluations.
    dead_pmt_rates: list of dead PMT rates
    iterations_per_rate: how many times you want to run evaluation per dead PMT rate
    classify: flag variable to tell if you want to do this for classificaiton model
    regress:  similarly, for regression model
    '''
    settings = utils()

    print('Doing multiple evaluations')

    # TODO: Modify these values.
    classify = 0
    regress  = 1

    # dead_pmt_rates = [0.0, 0.03, 0.05, 0.07, 0.1] # [0.03, 0.05, 0.1]
    dead_pmt_rates = [0.0, 0.03, .05, 0.10]
    iterations_per_rate = 15

    print('Dead pmt rates (%)', np.array(dead_pmt_rates) * 100)
    print(f'for {iterations_per_rate} iterations per percent')

    if classify:
        print('ML task: Classification')
        matrix = multi_evaluations_classification(settings, probs=dead_pmt_rates, itr=iterations_per_rate)
        print(matrix)
    if regress:
        print('ML task: Regression')
        matrix = multi_evaluations_regression(settings, probs=dead_pmt_rates, itr=iterations_per_rate,
                                              config_name='t2k_resnet_eval_dead',
                                              one_itr_for_zero_percent=True)
        print(matrix)