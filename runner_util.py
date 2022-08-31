import configparser
import random
import pickle
import os

import h5py

from WatChMaL.watchmal.model.classifier import Classifier, PassThrough
from WatChMaL.watchmal.model.pointnet import PointNetFeat
from WatChMaL.watchmal.model.resnet import resnet18
from WatChMaL.watchmal.model.classifier import PointNetFullyConnected
from WatChMaL.watchmal.dataset.t2k.t2k_dataset import PointNetT2KDataset, T2KCNNDataset

import torch
from torch.utils.data.sampler import SubsetRandomSampler


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
        for key in config[arch]:
            #use lower() to ignore any mistakes in capital letter in config file
            if 'InputPath'.lower() in key.lower():
                self.inputPath = config[arch][key]
            elif 'OutputPath'.lower() in key.lower():
                self.outputPath = config[arch][key]
            elif 'NetworkArchitecture'.lower() in key.lower():
                self.arch = config[arch][key]
            elif 'Classifier'.lower() in key.lower():
                self.classifier = config[arch][key]
            elif 'FeatureExtractor'.lower() in key.lower():
                self.featureExtractor = config[arch][key]
            elif 'DoClassification'.lower() in key.lower():
                self.doClassification = config[arch].getboolean(key)
            elif 'DoRegression'.lower() in key.lower():
                self.doRegression = config[arch].getboolean(key)
            elif 'UseGPU'.lower() in key.lower():
                self.useGPU = config[arch].getboolean(key)
            elif 'BatchSize'.lower() in key.lower():
                self.batchsize = config[arch].getint(key)
            elif 'Optimizer'.lower() in key.lower():
                self.optimizer = config[arch][key]
            elif 'NumClasses'.lower() in key.lower():
                self.numClasses = config[arch].getint(key)
            elif 'Epochs'.lower() in key.lower():
                self.epochs = config[arch].getint(key)
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
                self.trainTestSplit = config[arch].getfloat(key)
            elif 'TestValSplit'.lower() in key.lower():
                self.testValSplit = config[arch].getfloat(key)
            elif 'DataModel'.lower() in key.lower():
                self.dataModel = config[arch][key]
            elif 'PMTPositionsFile'.lower() in key.lower():
                self.pmtPositionsFile = config[arch][key]
            elif 'RestoreBestState'.lower() in key.lower():
                self.restoreBestState = config[arch].getboolean(key)
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

    def initClassifier(self):
        """Initializes the classifier and regression to be used in the main classification engine
        """
        #make a dictionary to avoid ugly array of if statements. Add lambda so that functions only get used if called in classification_engine below
        #use lower() to ignore any mistakes in capital letter in config file
        classifier_dictionary = {'PassThrough'.lower(): lambda : PassThrough(), 'PointNetFullyConnected'.lower(): lambda : PointNetFullyConnected(num_inputs=256, num_classes=self.numClasses)}
        regression_dictionary = {'resnet18'.lower(): lambda : resnet18(num_input_channels=1+int(self.useTime), num_output_channels=4), 'PointNetFeat'.lower(): lambda : PointNetFeat(k=4+int(self.useTime))}

        #Make sure to call () after every function because they are defined as lambdas in dictionary
        self.classification_engine = Classifier(regression_dictionary[self.featureExtractor.lower()](), classifier_dictionary[self.classifier.lower()](), self.numClasses) 
        if self.useGPU:
            gpu=0
            print("Running main worker function on device: {}".format(gpu))
            torch.cuda.set_device(gpu)
            self.classification_engine = self.classification_engine.cuda()

    def initDataset(self):
        """Initializes data_config and data_loader necessary to configure the training engine. Also sets up train/test/validation split of indices

        Returns:
            _type_: _description_
        """
        #dictionary to avoid if statements
        #use lower() to ignore any mistakes in capital letter in config file
        dataset_dictionary = {'T2KCNNDataset'.lower(): T2KCNNDataset, 'PointNetT2KDataset'.lower(): PointNetT2KDataset}
        data_config = {"dataset": self.inputPath.strip('\n'), "sampler":SubsetRandomSampler, "data_class": dataset_dictionary[self.dataModel.lower()]}
        #TODO: Smarter way to add architecture-dependent settings to data_config
        if 'ResNet'.lower() in self.arch.lower():
            data_config['pmt_positions_file'] = self.pmtPositionsFile
        if 'PointNet'.lower() in self.arch.lower():
            data_config['use_time'] = self.useTime
        data_loader = {"batch_size": self.batchsize, "num_workers":1}
        print(data_config)

        #Set up indices of train/test/val datasets using TrainTestSplit and TestValSplit from configuration settings
        length = len(h5py.File(self.inputPath.strip('\n'),mode='r')['event_hits_index'])
        indices = set(range(length))
        train =  set(random.sample(indices, int(self.trainTestSplit*len(list(indices)))))
        test = indices - train
        train_indices = list(train)
        other_indices = list(test)
        test_indices = other_indices[0:int(self.testValSplit*len(other_indices))]
        val_indices = other_indices[int(self.testValSplit*len(other_indices)):len(other_indices)]
        print(f'Train and Test sets share no indices: {set(train_indices).isdisjoint(test_indices)}')
        print(f'Train and Val sets share no indices: {set(train_indices).isdisjoint(val_indices)}')
        print(f'Test and Val sets share no indices: {set(test_indices).isdisjoint(val_indices)}')
        
        return data_config, data_loader, train_indices, test_indices, val_indices

    def initTrainConfig(self):
        """Additional configuration for training settings
        """
        self.train_config = train_config(self.epochs, self.reportInterval, self.valInterval, self.numValBatches, self.doCheckpointing, self.saveInterval)

    def initOptimizer(self):
        optimizer_dictionary = {"Adam".lower(): torch.optim.Adam}
        self.optimizer_engine = optimizer_dictionary[self.optimizer.lower()]
