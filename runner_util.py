import configparser

from WatChMaL.watchmal.model.classifier import Classifier, PassThrough
from WatChMaL.watchmal.model.pointnet import PointNetFeat
from WatChMaL.watchmal.model.resnet import resnet18
from WatChMaL.watchmal.model.classifier import PointNetFullyConnected


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
        self.initClassifier()

    def parser_string(self, config, arch):
        """Parses util_config.ini, converts strings to booleans/ints/float

        Args:
            config (_type_): The config data structure from the file given
            arch (_type_): The architecture chosen

        Returns:
            int: 0 if there is a problem
        """
        for key in config[arch]:
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
            else:
                print(f'Variable {key} not found, exiting')
                return 0

    def initClassifier(self):
        #make a dictionary to avoid ugly array of if statements. Add lambda so that functions only get used if called in classification_engine below
        classifier_dictionary = {'PassThrough'.lower(): lambda : PassThrough(), 'PointNetFullyConnected'.lower(): lambda : PointNetFullyConnected(num_inputs=256, num_classes=self.numClasses)}
        regression_dictionary = {'resnet18'.lower(): lambda : resnet18(num_input_channels=1+int(self.useTime), num_output_channels=4), 'PointNetFeat'.lower(): lambda : PointNetFeat(k=4+int(self.useTime))}

        #Make sure to call () after every function because they are defined as lambdas in dictionary
        self.classification_engine = Classifier(regression_dictionary[self.featureExtractor.lower()](), classifier_dictionary[self.classifier.lower()](), self.numClasses) 
        if self.useGPU:
            self.classification_engine = self.classification_engine.cuda()
