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
import re


#from analysis.classification import WatChMaLClassification
#from analysis.classification import plot_efficiency_profile
#from analysis.utils.plotting import plot_legend
import WatChMaL.analysis.utils.math as math

from analyze_output.analyze_regression import analyze_regression
from analyze_output.analyze_classification import analyze_classification

from runner_util import utils, analysisUtils, train_config, make_split_file
from WatChMaL.analysis.utils.binning import get_binning

# import torch
# from torchmetrics import AUROC, ROC

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

#from lxml import etree

import hydra

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--doTraining", help="run training", action="store_true")
parser.add_argument("--doFiTQun", help="run fitqun results", action="store_true")
parser.add_argument("--doEvaluation", help="run evaluation on already trained network", action="store_true")
parser.add_argument("--doMultiEvaluations", help="run evaluation on already trained network", action="store_true")
parser.add_argument("--debug", help="arg to run something", action="store_true")
parser.add_argument("--doComparison", help="run comparison", action="store_true")
parser.add_argument("--doQuickPlots", help="Make performance plots", action="store_true")
parser.add_argument("--doAnalysis", help="run analysis of ml and/or fitqun", action="store_true")
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


def get_performance_stats(settings, variable_list=[], variables=[]):
    # softmaxes = np.load(settings.outputPath+'/'+'softmax.npy')
    # labels = np.load(settings.outputPath+'/'+'labels.npy')

    softmaxes = np.load(args.evaluationOutputDir+'/'+'softmax.npy') # prediction
    labels = np.load(args.evaluationOutputDir+'/'+'labels.npy') # true
    
    '''
        0: e
        1: mu
        2: piplus
    '''

    # 1: muon-pip. Assume muon is 'signal(+)' and pip is 'background(-)'
    mu_p_mask = np.isin(labels, [1, 2])
    softmax_mu_p = softmaxes[mu_p_mask][:, [2,1]] # n by 2 matrix. 0th col = muon (+), 1st col = piplus (-)
    labels_mu_p  = labels[mu_p_mask]

    pred_prob_mu = softmax_mu_p[:, 0] / np.sum(softmax_mu_p, axis=1)

    fpr, tpr, _ = roc_curve(labels_mu_p, pred_prob_mu, pos_label=2)
    roc_auc_mu_p = auc(fpr, tpr)


    # 2: e-mu. Assume e = (+), and mu = (-)

    e_mu_mask = np.isin(labels, [0, 1])
    softmax_e_mu = softmaxes[e_mu_mask][:, [0,1]] # n by 2 matrix. 0th col = e (+), 1st col = mu (-)
    labels_e_mu  = labels[e_mu_mask]

    pred_prob_e = softmax_e_mu[:, 0] / np.sum(softmax_e_mu, axis=1)

    fpr, tpr, _ = roc_curve(labels_e_mu, pred_prob_e, pos_label=0)
    roc_auc_e_mu = auc(fpr, tpr)
    
    return np.array([roc_auc_mu_p, roc_auc_e_mu])
    
    
    
    # print(f'Unique labels in test set: {np.unique(labels,return_counts=True)}')
    # print('unique labels:', np.unique(labels))

    # print('prediction', softmaxes)
    # print('prediction', softmaxes.shape)

    # print('true', labels)
    # print('true', labels.shape)

    # print('2nd col of prediction:', softmaxes[:, 1].shape)
    # print('2nd col of prediction:', softmaxes[:, 1])

    # auc_ovr = roc_auc_score(labels, softmaxes, multi_class='ovr')
    # auc_ovo = roc_auc_score(labels, softmaxes, multi_class='ovo')
    # print(f'AUC: {auc}')
    # return np.array([auc_ovr, auc_ovo])

def run_evaluation(count=1, dead_pmt_seed=5, dead_pmt_rate=.03):
    print(f"run_evaluation(count={count}, dead_pmt_seed={dead_pmt_seed}, dead_pmt_rate={dead_pmt_rate})")


    settings = utils()
    settings.outputPath = args.evaluationOutputDir
    settings.set_output_directory()
    default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_classifier"] 
    indicesFile = check_list_and_convert(settings.indicesFile)
    perm_output_path = settings.outputPath

    default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_classifier"] 
    settings.outputPath = args.evaluationInputDir
    

    default_call.append("hydra.run.dir=" +str(args.evaluationInputDir))
    default_call.append("dump_path=" +str(args.evaluationOutputDir))
    # default_call.append("dump_path=" + dump_path)
    print(default_call)
    default_call += [f"data.dataset.dead_pmt_rate={dead_pmt_rate}", f"data.dataset.dead_pmt_seed={dead_pmt_seed}"]
    # for now just pmt rate

    # subprocess.call(default_call)
    result = subprocess.run(default_call, capture_output=True, text=True)
    # print(default_call)
    print(f'{default_call} was called. No problem. ')
    if result.stderr is not None:
        print("Error: ", result.stderr)
    print("Output: ", result.stdout)
    return result

def copy_npy(i, s, p):
    '''
    
    Parameters
    ----------
    i: iteration
    s: seed
    p: probability
        In [0, 1]
    '''
    date_time_str = datetime.today().strftime("%Y%m%d%H%M%S")
    dir_name = args.evaluationOutputDir + f'multiEval_seed_{s}_{i}th_itr_{round(p*100)}_percent_{date_time_str}/'
    os.makedirs(dir_name, exist_ok=True)
    source_dir = args.evaluationOutputDir
    # cp_command = f'cp {source_dir}/labels.npy {dir_name}'
    
    cp_command = ["cp", f"{source_dir}/indices.npy",f"{source_dir}/labels.npy",f"{source_dir}/softmax.npy", dir_name]

    result = subprocess.run(cp_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


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
    default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_classifier"] 
    indicesFile = check_list_and_convert(settings.indicesFile)
    perm_output_path = settings.outputPath

    default_call = ["python", "WatChMaL/main.py", "--config-name=t2k_resnet_eval_classifier"] 


    settings.outputPath = args.evaluationInputDir
    default_call.append("hydra.run.dir=" +str(args.evaluationInputDir))
    default_call.append("dump_path=" +str(args.evaluationOutputDir))
    print(default_call)
    subprocess.call(default_call)
    #end_training(settings)

if args.doMultiEvaluations:
    # a = np.ones((3, 5))
    # np.savetxt(args.evaluationOutputDir + 'avg_metrics_' + datetime.today().strftime("%Y%m%d%H%M%S") +'.csv', a, delimiter=',')
    settings = utils()
    # print(get_performance_stats(settings))

    # debug
    # print(get_performance_stats(settings))


    ###################


    # actual doMultiEval
    probs = [0.03, 0.05]
    itr = 3
    acc_loss = None
    aucs = None
    avg_metrics = None
    
    for prob in probs:
        sums = np.zeros((1, 4)) # loss, accuracy, auc, auc
        for i in range(itr):
            s = i # seed
            r1 = run_evaluation(i, s, prob)
            r2 = get_performance_stats(settings)
            # r2 = np.array([.9 + .1*s*prob, .85 + .12*s*prob])
            print(r1.stdout)
            print(r1.stderr)
            # accuracy and loss
            acc_loss_curr = re.findall(r'Average evaluation .*: (\d*\.\d*)', r1.stdout)
            # acc_loss_curr = np.array([0.5 +.1*s*prob, .7+.1*s*prob])
            acc_loss_curr = np.array([float(m) for m in acc_loss_curr])

            

            # acc_loss_curr = np.vstack(np.array([prob, s, s]), acc_loss_curr)
            
            if acc_loss is None:
                acc_loss = acc_loss_curr
            else:
                acc_loss = np.vstack((acc_loss, acc_loss_curr))

            # aucs
            # aucs_curr = np.vstack(np.array([prob, s, s]), r2)
            aucs_curr = r2
            if aucs is None:
                aucs = aucs_curr
            else:
                aucs = np.vstack((aucs, aucs_curr))
            
            sums += np.hstack((acc_loss_curr, aucs_curr))

            copy_npy(i, s, prob)
              
        print(f"[average loss, average accuracy, average AUC1, average AUC2] for dead prob of {prob} and {itr} iterations")
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
    
    np.savetxt(args.evaluationOutputDir + 'acc_loss_log_' + date_time_str +'.csv', acc_loss, delimiter=',')
    np.savetxt(args.evaluationOutputDir + 'auc_log_' + date_time_str +'.csv', aucs, delimiter=',')

if args.debug:
    # settings = utils()
    # a = get_performance_stats(settings)
    print("a")
    # call = 'mkdir ' + args.evaluationOutputDir + '/hello'
    # s = 1
    # i = 2
    # p = .035
    # dir_name = args.evaluationOutputDir + f'multiEval_seed_{s}_{i}th_itr_{round(p*100)}_percent/'
    # os.makedirs(dir_name, exist_ok=True)
    # source_dir = args.evaluationOutputDir
    # # cp_command = f'cp {source_dir}/labels.npy {dir_name}'
    # cp_command = ["cp", f"{source_dir}/indices.npy",f"{source_dir}/labels.npy",f"{source_dir}/softmax.npy", dir_name]

    # result = subprocess.run(cp_command, capture_output=True, text=True)
    # print(result.stdout)
    # print(result.stderr)
    # copy_npy(1, 2, .5)

    path_each = args.evaluationOutputDir + '/multiEval_seed_0_0th_itr_3_percent_20240529134944'

    softmaxes = np.load(path_each+'/'+'softmax.npy') # prediction
    labels = np.load(path_each+'/'+'labels.npy') # true
    
    '''
        0: e
        1: mu
        2: piplus
    '''

    # 1: muon-pip. Assume muon is 'signal(+)' and pip is 'background(-)'
    mu_p_mask = np.isin(labels, [1, 2])
    softmax_mu_p = softmaxes[mu_p_mask][:, [2,1]] # n by 2 matrix. 0th col = muon (+), 1st col = piplus (-)
    labels_mu_p  = labels[mu_p_mask]

    pred_prob_mu = softmax_mu_p[:, 0] / np.sum(softmax_mu_p, axis=1)

    fpr, tpr, _ = roc_curve(labels_mu_p, pred_prob_mu, pos_label=2)
    roc_auc_mu_p = auc(fpr, tpr)


    # 2: e-mu. Assume e = (+), and mu = (-)

    e_mu_mask = np.isin(labels, [0, 1])
    softmax_e_mu = softmaxes[e_mu_mask][:, [0,1]] # n by 2 matrix. 0th col = e (+), 1st col = mu (-)
    labels_e_mu  = labels[e_mu_mask]

    pred_prob_e = softmax_e_mu[:, 0] / np.sum(softmax_e_mu, axis=1)

    fpr2, tpr2, _ = roc_curve(labels_e_mu, pred_prob_e, pos_label=0)
    roc_auc_e_mu = auc(fpr2, tpr2)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(tpr, 1/fpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_mu_p)
    ax.set_yscale('log')
    # plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('Efficiency')
    plt.ylabel('Rejection')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('roc_curve.png')

    print("end")
    

if args.testParser:
    pass

if args.doAnalysis:
    settings = analysisUtils()
    settings.set_output_directory()

    if settings.doRegression:
        #analyze_regression(settings.inputPath, settings.mlPath, settings.fitqunPath, settings.particleLabel, settings.target, settings.outputPlotPath)
        analyze_regression(settings)
    if settings.doClassification:
        analyze_classification(settings)
    


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
