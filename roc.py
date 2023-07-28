import torch
import matplotlib.pyplot as plt
from compare_outputs import *
import argparse
import glob

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--doTraining", help="run training", action="store_true")
parser.add_argument("--doComparison", help="run comparison", action="store_true")
parser.add_argument("--doQuickPlots", help="Make performance plots", action="store_true")
parser.add_argument("--doIndices", help="create train/val/test indices file", action="store_true")
parser.add_argument("--testParser", help="run training", action="store_true")
parser.add_argument("--plotInput", help="run training")
parser.add_argument("--comparisonFolder", help="run training")
parser.add_argument("--numFolds", help="run training")
parser.add_argument("--indicesInput", help="run training")
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
                            '--numFolds','foo','@args_training.txt',
                            '--training_output_path','foo','@args_training.txt'])
class DoRoc():
    def __init__(self) -> None:
        self.list_of_directories = []

    def roc(self):
        print('1234')
        plot_folder = (self.directory + '/plots/')
        #dirs =  str(glob.glob(directory+'/*/'))
        self.set_output_directory(plot_folder)
        print('fdf')
        for i, dir in enumerate(self.list_of_directories):
            print('asf')
            self.set_output_directory(plot_folder+dir.replace(self.directory,''))
            plot_output = plot_folder+dir.replace(self.directory,'')+"/"
            softmaxes = np.load(+'softmax.npy')
            labels = np.load(dirs+'labels.npy')
            fpr, tpr, thresholds = roc(torch.tensor(softmaxes[:,1]), torch.tensor(labels))
            plot_tuple = plot_roc(fpr,tpr,thr,'Electron', 'Muon', fig_list=[0,1,2], plot_label=arch_name)
            fig.savefig(plot_output + 'log_test.png', format='png')
