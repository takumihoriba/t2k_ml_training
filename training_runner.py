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

from runner_util import utils

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

class train_config():
    def __init__(self,epochs, report_interval, val_interval, num_val_batches, checkpointing, save_interval) -> None:
            self.epochs=epochs
            self.report_interval=report_interval
            self.val_interval = val_interval
            self.num_val_batches = num_val_batches
            self.checkpointing = checkpointing
            self.save_interval = save_interval

if args.doTraining:

    num_files=1
    file_paths = args.training_input
    if ".txt" in args.training_input:
        print(f'Getting files from: {str(args.training_input)}')
        text_file = open(args.training_input, "r")
        file_paths = text_file.readlines()
        num_files = len(file_paths)

    do_resnet=True
    do_pointnet=False
    if do_resnet==True:
        do_pointnet=False

    gpu = 0 #TODO: Assign in utils
    print("Running main worker function on device: {}".format(gpu))
    torch.cuda.set_device(gpu)
    if do_pointnet:
        model = Classifier( PointNetFeat(k=5), PointNetFullyConnected(num_inputs=256, num_classes=3), num_classes=3).cuda() 
        engine = ClassifierEngine(model, 0, gpu, args.training_output_path)
        length = len(h5py.File(file_paths[0].strip('\n')+'combine_combine.hy',mode='r')['event_hits_index'])
        data_config = {"dataset": file_paths[0].strip('\n')+'combine_combine.hy', "sampler":SubsetRandomSampler, "data_class": PointNetT2KDataset, "use_positions": True, "use_time":True}
        data_loader = {"batch_size": 512, "num_workers":1}
    if do_resnet:
        model = Classifier(resnet18(num_input_channels=2, num_output_channels=4), PassThrough(5,3), num_classes=3).cuda()
        engine = ClassifierEngine(model, 0, gpu, args.training_output_path)
        length = len(h5py.File(file_paths[0].strip('\n')+'combine_combine.hy',mode='r')['event_hits_index'])
        data_config = {"dataset": file_paths[0].strip('\n')+'combine_combine.hy', "sampler":SubsetRandomSampler, "data_class": T2KCNNDataset, "pmt_positions_file": 'data/sk_wcsim_imagefile.npy'}
        data_loader = {"batch_size": 256, "num_workers":1}
    engine.configure_data_loaders(data_config, data_loader, False, 0, indices = length)
    engine.configure_optimizers(torch.optim.Adam)
    train_config_test = train_config(0, 50, 50 , 5, False, None)
    engine.train(train_config_test)

if args.testParser:
    settings = utils()

if args.doQuickPlots:
    fig = disp_learn_hist_smoothed(args.plotInput, losslim=2, show=False)
    fig.savefig(args.plotOutput+'resnet_test.png', format='png')
