import configparser
import random
import pickle
import os

from datetime import datetime

import h5py
import numpy as np

from sklearn.model_selection import KFold, train_test_split

from WatChMaL.watchmal.model.classifier import Classifier, PassThrough, PointNetFullyConnected, ResNetFullyConnected
from WatChMaL.watchmal.model.pointnet import PointNetFeat
from WatChMaL.watchmal.model.resnet import resnet18
#from WatChMaL.watchmal.dataset.t2k.t2k_dataset import PointNetT2KDataset, T2KCNNDataset

import torch
from torch.utils.data.sampler import SubsetRandomSampler

def calc_dwall_cut(file,cut):
    temp_x = h5py.File(file,mode='r')['positions'][:,0,0]
    temp_y = h5py.File(file,mode='r')['positions'][:,0,1]
    temp_r = np.sqrt(np.add(np.square(np.array(temp_x)), np.square(np.array(temp_y))))
    temp_z = np.abs(np.array(h5py.File(file,mode='r')['positions'][:,0,2]))
    return np.logical_and(temp_r < (1690-cut), temp_z < (1850-cut))


def make_split_file(h5_file,train_val_test_split=[0.70,0.15], output_path='data/', seed=0, nfolds=3):
    """Outputs indices to split h5 files into train/test/val 

    Args:
        h5_file (_type_): path+name of h5 file (combination of many root files)
        train_val_test_split (list, optional): Train and Val split. Test is assumed as 1-train-val. Defaults to [0.70,0.15]. Only used if nfolds is 1
        output_path (str, optional): where to store the index file. Defaults to 'data/'.
        seed (int, optional): Seed. Leave at 0 unless you know what you're doing. Defaults to 0.
        nfolds (int, optional): Number of folds. Makes it so that you can make different folds for n fold validation. Defaults to 3.
    """
    #Check if you can actually do the number of folds requested
    nfolds = int(nfolds)
    print(f'nfolds: {nfolds}, nfolds type: {type(nfolds)}')
    """
    if (1.0-train_val_test_split[0]-train_val_test_split[1])*int(nfolds) > 1:
        print(f"ERROR: {nfolds} folds is too many for the test proportion ({(1.0-train_val_test_split[0]-train_val_test_split[1])}) requested ")
        return 0
    """

    length = len(h5py.File(h5_file,mode='r')['event_hits_index'])
    unique_root_files, unique_inverse, unique_counts = np.unique(h5py.File(h5_file,mode='r')['root_files'], return_inverse=True, return_counts=True)
    dwall_cut_value = 0
    print(f'WARNING: Applying a dwall cut of {dwall_cut_value} cm')
    dwall_cut = calc_dwall_cut(h5_file, dwall_cut_value)
    print(f'WARNING: Removing veto events')
    print(f'WARNING: Removing range=-999 events')
    print('WARNING: Removing events with decay electrons')
    print(f'Original # indices: {len(dwall_cut)}')
    #Range and decay electron
    #indices_to_keep = np.array(range(len(dwall_cut)))[np.logical_and(np.logical_and(np.logical_and(dwall_cut,~h5py.File(h5_file,mode='r')['veto'][:]), np.ravel(h5py.File(h5_file,mode='r')['primary_charged_range']) != -999), np.ravel(h5py.File(h5_file,mode='r')['decay_electron_exists'][:]==0))]
    #indices_to_keep = np.array(range(len(dwall_cut)))[np.logical_and(np.logical_and(dwall_cut,~h5py.File(h5_file,mode='r')['veto'][:]), np.ravel(h5py.File(h5_file,mode='r')['primary_charged_range']) != -999)]
    #indices_to_keep = np.array(range(len(dwall_cut)))[np.logical_and(np.logical_and(dwall_cut,np.logical_and(~h5py.File(h5_file,mode='r')['veto'][:], ~h5py.File(h5_file,mode='r')['decay_electron_exists'][:])), np.ravel(h5py.File(h5_file,mode='r')['primary_charged_range']) != -999)]
    #indices_to_keep = np.array(range(len(dwall_cut)))[np.logical_and(dwall_cut, ~h5py.File(h5_file,mode='r')['veto'][:])]
    
    with h5py.File(h5_file, mode='r') as h5fw:
        # select indices only with 'keep_event' == True (if key exists), instead of keeping all events
        if 'keep_event' in h5fw.keys():
            print(f'NEW! WARNING: Removing additional events to flatten truth visible energy distribution')
            keep_bool = np.array(h5fw['keep_event'])
            indices_to_keep = np.where(keep_bool == True)[0] 
        #Keep all    
        else:
            indices_to_keep = np.array(range(len(dwall_cut)))
            
    print(f'indices to keep: {len(indices_to_keep)}')
    #Based on root files, divide indices into train/val/test
    length_rootfiles = len(unique_root_files)

    if nfolds==1:
        random.seed(seed)

        train_rootfile_indices = random.sample(range(length_rootfiles), int(train_val_test_split[0]*len(list(range(length_rootfiles)))))
        train_indices = np.isin(unique_inverse, train_rootfile_indices)
        train_indices = np.array(range(length), dtype='int64')[train_indices]
        train_rootfiles_set = set(train_rootfile_indices)
        index_rootfiles_set = set(range(length_rootfiles))
        other_rootfiles_indices = list(index_rootfiles_set - train_rootfiles_set)
        val_rootfile_indices = other_rootfiles_indices[0:int(train_val_test_split[1]*len(other_rootfiles_indices))]
        test_rootfile_indices = other_rootfiles_indices[int(train_val_test_split[1]*len(other_rootfiles_indices)):len(other_rootfiles_indices)] 
        test_indices = np.isin(unique_inverse, test_rootfile_indices)
        test_indices = np.array(range(length), dtype='int64')[test_indices]
        val_indices = np.isin(unique_inverse, val_rootfile_indices)
        val_indices = np.array(range(length), dtype='int64')[val_indices]

        #Applying cuts
        train_indices = train_indices[np.isin(train_indices, indices_to_keep)]
        labels = h5py.File(h5_file,mode='r')['labels']



        test_indices = test_indices[np.isin(test_indices, indices_to_keep)]
        val_indices = val_indices[np.isin(val_indices, indices_to_keep)]
        print(np.unique(labels[train_indices],return_counts=True))
        print(np.unique(labels[val_indices],return_counts=True))
        print(np.unique(labels[test_indices],return_counts=True))

        np.savez(output_path + 'train'+str(train_val_test_split[0])+'_val'+str(train_val_test_split[1])+'_test'+str(1-train_val_test_split[0]-train_val_test_split[1])+'.npz',
                    test_idxs=test_indices, val_idxs=val_indices, train_idxs=train_indices)
    else:
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        for i, (train_rootfile_indices, test_index) in enumerate(kf.split(range(length_rootfiles))):
            split = train_test_split(test_index, train_size=0.5, shuffle=True, random_state=seed)
            val_rootfile_indices = split[0]
            test_rootfile_indices = split[1]
            train_indices = np.isin(unique_inverse, train_rootfile_indices)
            train_indices = np.array(range(length), dtype='int64')[train_indices]
            print(train_indices)
            test_indices = np.isin(unique_inverse, test_rootfile_indices)
            test_indices = np.array(range(length), dtype='int64')[test_indices]
            val_indices = np.isin(unique_inverse, val_rootfile_indices)
            val_indices = np.array(range(length), dtype='int64')[val_indices]

            #Applying cuts
            train_indices = train_indices[np.isin(train_indices, indices_to_keep)]
            test_indices = test_indices[np.isin(test_indices, indices_to_keep)]
            val_indices = val_indices[np.isin(val_indices, indices_to_keep)]

            labels = h5py.File(h5_file,mode='r')['labels']


            print(f"Fold {i}")
            print(np.unique(np.ravel(labels)[train_indices], return_counts=True))
            print(np.unique(np.ravel(labels)[val_indices],return_counts=True))
            print(np.unique(np.ravel(labels)[test_indices],return_counts=True))
            print(output_path)
            np.savez(output_path + 'train_val_test_nFolds'+str(nfolds)+'_fold'+str(i)+'.npz',
                    test_idxs=test_indices, val_idxs=val_indices, train_idxs=train_indices)


class train_config():
    def __init__(self,epochs, report_interval, val_interval, num_val_batches, checkpointing, save_interval) -> None:
        self.epochs=epochs
        self.report_interval=report_interval
        self.val_interval = val_interval
        self.num_val_batches = num_val_batches
        self.checkpointing = checkpointing
        self.save_interval = save_interval

class utils():
    """Utility class to read in config file, prepare WatChMaL training
    """
    def __init__(self, parser_file='util_config.ini') -> None:
        config = configparser.ConfigParser()
        config.read(parser_file)
        arch = config['DEFAULT']['NetworkArchitecture'] 
        self.parser_string(config, arch)

    def parser_string(self, config, arch):
        """Parses util_config.ini, converts strings to booleans/ints/float

        Args:
            config (_type_): The config data structure from the file given
            arch (_type_): The architecture chosen

        Returns:
            int: 0 if there is a problem
        """
        self.list_for_sweep = []
        for key in config[arch]:
            #use lower() to ignore any mistakes in capital letter in config file
            if 'InputPath'.lower() in key.lower():
                self.inputPath = config[arch][key]
            elif 'IndicesFile'.lower() in key.lower():
                if ',' in config[arch][key]:
                    self.indicesFile = self.getListOfInput(config[arch][key], str)
                    self.list_for_sweep.append(self.indicesFile)
                else:
                    self.indicesFile = config[arch][key]
            elif 'OutputPath'.lower() in key.lower():
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y-%H%M%S")
                output_file = config[arch][key]
                self.outputPath = output_file
            elif 'NetworkArchitecture'.lower() in key.lower():
                self.arch = config[arch][key]
            elif 'Classifier'.lower() in key.lower():
                self.classifier = config[arch][key]
            elif 'FeatureExtractor'.lower() in key.lower():
                if ',' in config[arch][key]:
                    self.featureExtractor = self.getListOfInput(config[arch][key], str)
                    self.list_for_sweep.append(self.featureExtractor)
                else:
                    self.featureExtractor = config[arch][key]
            elif 'DoClassification'.lower() in key.lower():
                self.doClassification = config[arch].getboolean(key)
            elif 'DoRegression'.lower() in key.lower():
                self.doRegression = config[arch].getboolean(key)
            elif 'UseGPU'.lower() in key.lower():
                self.useGPU = config[arch].getboolean(key)
            elif 'GPUNumber'.lower() in key.lower():
                if len(config[arch][key]) > 1:
                    self.getGPUNumber(config[arch][key])
                else:
                    self.gpuNumber = config[arch].getint(key)
                    self.multiGPU = False
            elif 'TrainBatchSize'.lower() in key.lower():
                self.TrainBatchSize = config[arch].getint(key)
            elif 'ValBatchSize'.lower() in key.lower():
                self.ValBatchSize = config[arch].getint(key)
            elif 'Optimizer'.lower() in key.lower():
                self.optimizer = config[arch][key]
            elif 'NumClasses'.lower() in key.lower():
                self.numClasses = config[arch].getint(key)
            elif 'Epochs'.lower() in key.lower():
                self.epochs = config[arch].getint(key)
            elif 'KernelSize'.lower() in key.lower():
                if ',' in config[arch][key]:
                    self.kernel = self.getListOfInput(config[arch][key], int)
                    self.list_for_sweep.append(self.kernel)
                else:
                    self.kernel = config[arch][key]
            elif 'Stride'.lower() in key.lower():
                self.stride = config[arch].getint(key)
                if ',' in config[arch][key]:
                    self.stride = self.getListOfInput(config[arch][key], int)
                    self.list_for_sweep.append(self.stride)
                else:
                    self.stride = config[arch][key]
            elif 'ReportInterval'.lower() in key.lower():
                self.reportInterval = config[arch].getint(key)
            elif 'ValInterval'.lower() in key.lower():
                self.valInterval = config[arch].getint(key)
            elif 'NumValBatches'.lower() in key.lower():
                self.numValBatches = config[arch].getint(key)
            elif 'DoCheckpointing'.lower() in key.lower():
                self.doCheckpointing = config[arch].getboolean(key)
            elif 'SaveInterval'.lower() in key.lower():
                if 'None' in config[arch][key]:
                    self.saveInterval = None
                else:
                    self.saveInterval = config[arch][key]
            elif 'UseTime'.lower() in key.lower():
                self.useTime = config[arch].getboolean(key)
            elif 'TrainTestSplit'.lower() in key.lower():
                self.train_val_test_split = config[arch].getfloat(key)
            elif 'TestValSplit'.lower() in key.lower():
                self.testValSplit = config[arch].getfloat(key)
            elif 'DataModel'.lower() in key.lower():
                self.dataModel = config[arch][key]
            elif 'PMTPositionsFile'.lower() in key.lower():
                self.pmtPositionsFile = config[arch][key]
            elif 'RestoreBestState'.lower() in key.lower():
                self.restoreBestState = config[arch].getboolean(key)
            elif 'LearningRateDecay'.lower() in key.lower():
                if ',' in config[arch][key]:
                    self.lr_decay = self.getListOfInput(config[arch][key], float)
                    self.list_for_sweep.append(self.lr_decay)
                else:
                    self.lr_decay = config[arch].getfloat(key)
            elif 'LearningRate'.lower() in key.lower():
                if ',' in config[arch][key]:
                    self.lr = self.getListOfInput(config[arch][key], float)
                    self.list_for_sweep.append(self.lr)
                else:
                    self.lr = config[arch].getfloat(key)
            elif 'WeightDecay'.lower() in key.lower():
                if ',' in config[arch][key]:
                    self.weightDecay = self.getListOfInput(config[arch][key], float)
                    self.list_for_sweep.append(self.weightDecay)
                else:
                    self.weightDecay = config[arch].getfloat(key)
            elif 'Seed'.lower() in key.lower():
                self.seed = config[arch].getint(key)
            else:
                print(f'Variable {key} not found, exiting')
                return 0

    def save_options(self,filepath,filename):
        """Save the class and its variables in file

        Args:
            filepath (_type_): Path to file
            filename (_type_): Name of file
        """
        with open(filepath+'/'+filename,'wb') as f:
            pickle.dump(self,f)

    def load_options(self,filepath,filename):
        """Load the class and its variables from file

        Args:
            filepath (_type_): Path to file
            filename (_type_): Name of file

        Returns:
            WCSimOptions class object: Loaded class
        """
        with open(filepath+'/'+filename,'rb') as f:
            new_options = pickle.load(f)
            return new_options

    def set_output_directory(self):
        """Makes an output file as given in the arguments
        """
        if not(os.path.exists(self.outputPath) and os.path.isdir(self.outputPath)):
            try:
                os.makedirs(self.outputPath)
            except FileExistsError as error:
                print("Directory " + str(self.outputPath) +" already exists")
                if self.batch is True:
                    exit

    def getGPUNumber(self, gpu_input):
        numbers = gpu_input.split(",")
        numbers = list(map(int, numbers))
        self.gpuNumber = numbers
        self.multiGPU = True

    def getListOfInput(self, list_of_inputs, type):
        inputs = list_of_inputs.split(",")
        inputs = list(map(type, inputs))
        return inputs

    def checkLabels(self):
        with h5py.File(self.inputPath,mode='r') as h5fw:
            min_label = np.amin(h5fw['labels'])
            self.minLabel = min_label

    def initClassifier(self, kernel_size, stride):
        """Initializes the classifier and regression to be used in the main classification engine
        """
        #make a dictionary to avoid ugly array of if statements. Add lambda so that functions only get used if called in classification_engine below
        #use lower() to ignore any mistakes in capital letter in config file
        print('use time 1:', self.useTime)
        classifier_dictionary = {'PassThrough'.lower(): lambda : PassThrough(), 'PointNetFullyConnected'.lower(): lambda : PointNetFullyConnected(num_inputs=256, num_classes=self.numClasses)}
        regression_dictionary = {'resnet18'.lower(): lambda : resnet18(num_input_channels=1+int(self.useTime), num_output_channels=self.numClasses, conv1_kernel = kernel_size, conv1_stride = stride), 'PointNetFeat'.lower(): lambda : PointNetFeat(k=4+int(self.useTime))}

        #Make sure to call () after every function because they are defined as lambdas in dictionary
        self.classification_engine = Classifier(regression_dictionary[self.featureExtractor.lower()](), classifier_dictionary[self.classifier.lower()](), self.numClasses) 
        if self.useGPU:
            gpu=self.gpuNumber
            print("Running main worker function on device: {}".format(gpu))
            torch.cuda.set_device(gpu)
            self.classification_engine = self.classification_engine.cuda()

    def initDataset(self, rank):
        print("DOING INIT DATASET")
        """Initializes data_config and data_loader necessary to configure the training engine. Also sets up train/test/validation split of indices

        Returns:
            data_config: Dictionary of options containing data settings
            data_loader: Dictionary of options for loading the data
            train_indices: Indices of the dataset used for training
            test_indices: Indices of the dataset used for testing
            val_indices: Indices of the dataset used for validationg
        """
        #dictionary to avoid if statements
        #use lower() to ignore any mistakes in capital letter in config file
        dataset_dictionary = {'T2KCNNDataset'.lower(): T2KCNNDataset, 'PointNetT2KDataset'.lower(): PointNetT2KDataset}
        data_config = {"dataset": self.inputPath.strip('\n'), "sampler":SubsetRandomSampler, "data_class": dataset_dictionary[self.dataModel.lower()], "is_distributed": self.multiGPU}
        #TODO: Smarter way to add architecture-dependent settings to data_config
        if 'ResNet'.lower() in self.arch.lower():
            data_config['pmt_positions_file'] = self.pmtPositionsFile
        if 'PointNet'.lower() in self.arch.lower():
            data_config['use_time'] = self.useTime
        train_data_loader = {"batch_size": self.TrainBatchSize, "num_workers":4}
        val_data_loader = {"batch_size": self.ValBatchSize, "num_workers":4}
        print('use time 2:', self.useTime)

        #Set up indices of train/test/val datasets using TrainTestSplit and TesstValSplit from configuration settings
        random.seed(a=self.seed)

        length = len(h5py.File(self.inputPath.strip('\n'),mode='r')['event_hits_index'])
        unique_root_files, unique_inverse, unique_counts = np.unique(h5py.File(self.inputPath.strip('\n'),mode='r')['root_files'], return_inverse=True, return_counts=True)

        #Based on root files, divide indices into train/val/test
        length_rootfiles = len(unique_root_files)
        train_rootfile_indices = random.sample(range(length_rootfiles), int(train_val_test_split[0]*len(list(range(length_rootfiles)))))
        train_indices = np.isin(unique_inverse, train_rootfile_indices)
        train_indices = np.array(range(length))[train_indices]
        train_rootfiles_set = set(train_rootfile_indices)
        index_rootfiles_set = set(range(length_rootfiles))
        other_rootfiles_indices = list(index_rootfiles_set - train_rootfiles_set)
        test_rootfile_indices = other_rootfiles_indices[0:int(self.testValSplit*len(other_rootfiles_indices))]
        val_rootfile_indices = other_rootfiles_indices[int(self.testValSplit*len(other_rootfiles_indices)):len(other_rootfiles_indices)] 
        test_indices = np.isin(unique_inverse, test_rootfile_indices)
        test_indices = np.array(range(length))[test_indices]
        val_indices = np.isin(unique_inverse, val_rootfile_indices)
        val_indices = np.array(range(length))[val_indices]

        print(f'Train and Test sets share no indices: {set(train_indices).isdisjoint(test_indices)}')
        print(f'Train and Val sets share no indices: {set(train_indices).isdisjoint(val_indices)}')
        print(f'Test and Val sets share no indices: {set(test_indices).isdisjoint(val_indices)}')

        test_rootfiles = np.unique(np.array(unique_root_files[unique_inverse])[test_indices])
        if rank==0:
            print("Saving Test Rootfies...")
            np.save(self.outputPath + "test_rootfiles.npy", test_rootfiles)
        
        return data_config, train_data_loader, val_data_loader, train_indices, test_indices, val_indices

    def initTrainConfig(self):
        """Additional configuration for training settings
        """
        self.train_config = train_config(self.epochs, self.reportInterval, self.valInterval, self.numValBatches, self.doCheckpointing, self.saveInterval)

    def initOptimizer(self):
        optimizer_dictionary = {"Adam".lower(): torch.optim.Adam}
        self.optimizer_engine = optimizer_dictionary[self.optimizer.lower()]

    def getPlotInfo(self):
        return self.outputPath, self.arch
