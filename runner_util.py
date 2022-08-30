import configparser

class utils():
    def __init__(self, parser_file='util_config.ini') -> None:
        config = configparser.ConfigParser()
        config.read(parser_file)
        arch = config['DEFAULT']['NetworkArchitecture'] 
        self.parser_string(config, arch)

    def parser_string(self, config, arch):
        for key in config[arch]:
            if 'InputPath'.lower() in key.lower():
                self.inputPath = config[arch][key]
            if 'OutputPath'.lower() in key.lower():
                self.outputPath = config[arch][key]
            if 'NetworkArchitecture'.lower() in key.lower():
                self.arch = config[arch][key]
            if 'Classifier'.lower() in key.lower():
                self.classifier = config[arch][key]
            if 'FeatureExtractor'.lower() in key.lower():
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
