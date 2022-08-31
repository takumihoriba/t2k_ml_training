import argparse
import debugpy
import h5py
import logging

import torch

from WatChMaL.watchmal.model.classifier import Classifier, PassThrough
from WatChMaL.watchmal.model.pointnet import PointNetFeat
from WatChMaL.watchmal.model.resnet import resnet18
from WatChMaL.watchmal.model.classifier import PointNetFullyConnected
from WatChMaL.watchmal.engine.engine_classifier import ClassifierEngine
from WatChMaL.watchmal.dataset.t2k.t2k_dataset import PointNetT2KDataset, T2KCNNDataset
from WatChMaL.analysis.plot_utils import disp_learn_hist_smoothed

from torch.utils.data.sampler import SubsetRandomSampler

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


if args.doTraining:

    #Choose settings for utils class in util_config.ini
    settings = utils()
    if settings==0:
        print("Settings did not initialize properly, exiting...")
        exit


    #Initialize training configuration, classifier and dataset for engine training
    settings.initTrainConfig()
    settings.initClassifier()
    settings.initOptimizer()
    data_config, data_loader, train_indices, test_indices, val_indices = settings.initDataset()
    model = settings.classification_engine
    engine = ClassifierEngine(model, 0, 0, settings.outputPath)

    engine.configure_data_loaders(data_config, data_loader, False, 0, train_indices, test_indices, val_indices, settings)
    engine.configure_optimizers(settings.optimizer_engine)
    settings.set_output_directory()
    settings.save_options(settings.outputPath, 'training_settings')
    engine.train(settings)

if args.testParser:
    settings = utils()

if args.doQuickPlots:
    fig = disp_learn_hist_smoothed(args.plotInput, losslim=2, show=False)
    fig.savefig(args.plotOutput+'resnet_test.png', format='png')
