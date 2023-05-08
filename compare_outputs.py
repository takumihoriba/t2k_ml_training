import glob
import os
import numpy as np

import xml.etree.ElementTree as ET

from plotting import efficiency_plots

class dealWithOutputs():
    def __init__(self, directory) -> None:
        self.directory = directory
        self.list_of_directories = []
        self.list_of_indices_file = []
        self.list_of_input_variables = []
        self.list_of_output_stats = []
        self.list_of_input_files = []
    
    def add_output(self, directory, indices_file, inputFile, input_variables, output_stats):
        self.list_of_directories.append(directory)
        self.list_of_indices_file.append(indices_file)
        self.list_of_input_variables.append(input_variables)
        self.list_of_output_stats.append(output_stats)
        self.list_of_input_files.append(inputFile)

    def find_unique(self):
        set_of_same = []
        #Find which dictionaries are the same
        for i, item_1 in enumerate(self.list_of_input_variables):
            for j, item_2 in enumerate(self.list_of_input_variables):
                if item_1 == item_2 and i!=j:
                    set_of_same.append([i,j])
        
        out = []
        #Combine all same dictionaries into unique sets
        while len(set_of_same)>0:
            first, *rest = set_of_same
            first = set(first)

            lf = -1
            while len(first)>lf:
                lf = len(first)

                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r)))>0:
                        first |= set(r)
                    else:
                        rest2.append(r)     
                rest = rest2

            out.append(first)
            set_of_same = rest

        self.unique_sets = out


    def calc_stats(self):
        self.final_stats_dict = []
        self.final_variable_dict = []
        for set in self.unique_sets:
            temp_stats = {}
            for index in list(set):
                for stat in self.list_of_output_stats[index]:
                    if stat in temp_stats:
                        temp_stats[stat].append(float(self.list_of_output_stats[index][stat]))
                    else:
                        temp_stats[stat] = [float(self.list_of_output_stats[index][stat])]
            for stat in temp_stats:
                temp_stats[stat] = [np.mean(temp_stats[stat]), np.std(temp_stats[stat])]
            self.final_stats_dict.append(temp_stats)
            self.final_variable_dict.append(self.list_of_input_variables[list(set)[0]])

        print(self.final_variable_dict)
        print(self.final_stats_dict)

        #Assumes all dictionaries have same variables
        '''
        for key in self.final_variable_dict[0]:
            learning_rates = [d[key] for d in self.final_variable_dict]
            resultset = [key_2 for key_2, value in self.final_variable_dict[0].items() if key not in key_2]
            for i, items in self.final_variable_dict:
                label = ''
                for var in resultset:
                    pass
                    

            print(learning_rates)
            print(resultset)
        '''
        
    def set_output_directory(self, path):
        """Makes an output file as given in the arguments
        """
        if not(os.path.exists(path) and os.path.isdir(path)):
            try:
                os.makedirs(path)
            except FileExistsError as error:
                print("Directory " + str(path) +" already exists")

    def convert_variables_to_label(self,variable_dict):
        output_string = ''
        for i, key in enumerate(variable_dict):
            if i > 0:
                output_string = output_string + '\n'
            output_string = output_string + key +': ' + variable_dict[key]
        return output_string

    def make_plots(self):

        plot_folder = self.directory + '/plots/'
        self.set_output_directory(plot_folder)
        for i, dir in enumerate(self.list_of_directories):
            self.set_output_directory(plot_folder+dir.replace(self.directory,''))
            plot_output = plot_folder+dir.replace(self.directory,'')+"/"
            label = self.convert_variables_to_label(self.list_of_input_variables[i])
            run = efficiency_plots(self.list_of_input_files[i], '', dir, plot_output, label=label)
            fig,ax1= run.plot_training_progression(y_loss_lim=[0.,2.], doAccuracy=False, label=label)
            fig.tight_layout(pad=2.0) 
            fig.savefig(plot_output + 'log_test.png', format='png')

        



def compare_outputs(folder):
    dirs =  glob.glob(folder+'/*')
    outputs = dealWithOutputs(folder)
    for dir in dirs:
        #Checks if training is done by checking if training_stats.xml is output
        try:
            tree = ET.parse(dir+'/training_stats.xml')
        except FileNotFoundError:
            continue
        root = tree.getroot()
        input_variables = {}
        output_stats = {}
        indices_file = {}
        input_file = ''
        for child in root:
            if 'Variables' in child.tag:
                for child_2 in child:
                    if 'indicesFile' in child_2.tag:
                        indices_file[child_2.tag] = child_2.attrib['var']
                    else:
                        input_variables[child_2.tag] = child_2.attrib['var']
            if 'Stats' in child.tag:
                for child_2 in child:
                    output_stats[child_2.tag] = child_2.attrib['var']
            if 'Files' in child.tag:
                for child_2 in child:
                    if 'inputPath' in child_2.tag:
                        inputFile = child_2.attrib['var']
        outputs.add_output(dir, indices_file, inputFile, input_variables, output_stats)
    outputs.find_unique()
    outputs.calc_stats()
    outputs.make_plots()
