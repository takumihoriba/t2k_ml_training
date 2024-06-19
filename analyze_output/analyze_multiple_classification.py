import numpy as np

from analyze_output.utils.plotting import softmax_plots
from analyze_output.utils.math import get_cherenkov_threshold

import os
import random
import pandas as pd

import h5py

# from analysis.classification import plot_rocs2

from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile, plot_rocs
from analysis.classification import compute_AUC, compute_ROC

from analysis.utils.plotting import plot_legend
from analysis.utils.binning import get_binning
from analysis.utils.fitqun import read_fitqun_file, make_fitqunlike_discr, get_rootfile_eventid_hash, plot_fitqun_comparison
import analysis.utils.math as math

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


from scipy.optimize import curve_fit
from scipy.stats.mstats import mquantiles_cimj

import WatChMaL.analysis.utils.fitqun as fq

from tqdm import tqdm

from math import log10, floor, ceil

from sklearn import metrics


class MultiRegressionAnalysis:
    def __init__(self, settings, sub_dir_names, percents):
        """
        """
        self.settings = settings
        self.base_path = settings.mlPath
        self.percents = percents
        self.sub_dirs = sub_dir_names
        self.plot_counter = 0
        self.computed = False

        self.colors = ['blue', 'g', 'r', 'violet', 'k', 'c', 'm', 'orange', 'purple', 'brown']

        self.summary_stats_header = ["Dead PMT Rate (%)", "Count", "Min", "Q1", "Median", "Mean", "Q3", "Max", "SD"]

        self.set_ROC_info()
    
    def set_ROC_info(self):

        settings = self.settings

        if type(settings.signalLabels) == list:
            signal_label = [settings.signalLabels] # previously e_label
        else:
            signal_label = [settings.signalLabels]
        if type(settings.bkgLabels) == list:
            background_labels = settings.bkgLabels # previously mu_label
        else:
            background_labels = [settings.bkgLabels]
        

        label_names = ['Muon', 'Electron', 'Pion'] # The labels are muons 0, electrons 1, and pions 2

        self.signal_label_desc = label_names[signal_label[0]]
        self.background_labels_desc = 'Others' if len(background_labels) > 1 else label_names[background_labels[0]]

        self.signal_label = signal_label
        self.background_labels = background_labels

        self.roc_desc = f'{self.signal_label_desc}_vs_{self.background_labels_desc}'

    def analyze(self, tasks=['roc']):
        '''
        Analyzes the output of the multi-regression model
        tasks: list of strings
            list of tasks to perform
        '''
        for t in tasks:
            if t == 'roc':
                self.plot_mean_ROCs()
            elif t == 'roc_zoom':
                self.plot_mean_ROCs(xlim=[0.5, 1.03], ylim=None)
        return
    
    def plot_mean_ROCs(self, xlim=None, ylim=None):
        '''
        Plots the mean ROCs for the evaluations with different rates of dead PMTs
        Sets attributes on the way to keep mean ROC curves and AUCs
        xlim: list of 2 floats
            the lower and upper bounds of the x-axis
        ylim: list of 2 floats
            the lower and upper bounds of the y-axis
        '''

        if self.computed == False:
            self.compute_mean_ROCs()

        fig, ax = plt.subplots()

        # y_up_bd = 0
        # y_padding = 100

        for i, p in enumerate(list(set(self.percents))):
            ax.plot(self.base_tpr, self.mean_roc_curves[p], self.colors[i], label = f'Mean ROC({p}% Dead, AUC={round(self.mean_aucs[p], 4)})')
            # y_up_bd = max(max(self.mean_roc_curves[p]), y_up_bd)
        
        ax.set_yscale('log')
        ax.set_xlabel(f'{self.signal_label_desc} Tagging Efficiency')
        ax.set_ylabel(f'{self.background_labels_desc} Rejection')
        roc_title = self.roc_desc.replace('_', ' ')
        ax.set_title(f'{roc_title} Mean ROCs by Dead PMT Rates (%)')
        ax.legend()
        
        if xlim is not None:
            ax.set_xlim(xlim)

        # ax.set_ylim([0.01, y_up_bd])
        if ylim is not None:
            ax.set_ylim(ylim)

        fig.savefig(self.settings.outputPlotPath + f'/{self.roc_desc}_ROCs_mean_all_auc_{self.plot_counter}.png', format='png')
        self.plot_counter += 1

        
    def compute_mean_ROCs(self):
        '''
        Computes the mean ROCs for the evaluations with different rates of dead PMTs
        Sets following attributes:
            self.roc_curves_dict
            self.auc_dict
            self.base_tpr
            self.mean_roc_curves
            self.colors
            self.mean_aucs
            self.computed (to True)
        '''

        # base_path + sub_dirs[i] is where result of i-th evaluation is stored.
        base_path = self.settings.mlPath
        sub_dirs = self.sub_dirs
        percents = self.percents
        settings = self.settings

        colors = self.colors
        
        
        ##########
        
        roc_curves_dict = {}
        auc_dict = {}
        for p in list(set(percents)):
            roc_curves_dict[p] = []
            auc_dict[p] = []

        for i, sub_dir in enumerate(sub_dirs):
            roc_curves = []

            eval_output_path = base_path + sub_dir

            idx = np.array(sorted(np.load(str(eval_output_path) + "/indices.npy")))
            idx = np.unique(idx)
            softmax = np.array(np.load(str(eval_output_path) + "/softmax.npy"))
            
            labels_test = np.array(np.load(str(eval_output_path) + "/labels.npy"))

            # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
            hy = h5py.File(settings.inputPath + "/combine_combine.hy", "r")
            print(hy["labels"].shape)
            print(np.amax(idx))
            angles = np.array(hy['angles'])[idx].squeeze() 
            labels = np.array(hy['labels'])[idx].squeeze() 
            veto = np.array(hy['veto'])[idx].squeeze()
            energies = np.array(hy['energies'])[idx].squeeze()
            positions = np.array(hy['positions'])[idx].squeeze()
            #positions=true_positions_array.squeeze()
            directions = math.direction_from_angles(angles)
            rootfiles = np.array(hy['root_files'])[idx].squeeze()
            event_ids = np.array(hy['event_ids'])[idx].squeeze()
            #positions_ml = positions_array.squeeze()

            # calculate number of hits 
            events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
            nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()

            #Save ids and rootfiles to compare to fitqun, after applying cuts
            ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)

            softmax_sum = np.sum(softmax,axis=1)
            print(f"SOFTMAX SUM: {np.amin(softmax_sum)}")

            # calculate additional parameters 
            towall = math.towall(positions, angles, tank_axis = 2)
            ml_cheThr = list(map(get_cherenkov_threshold, labels))

            # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
            nhit_cut = nhits > 0 #25
            towall_cut = towall > 100
            # veto_cut = (veto == 0)
            hy_electrons = (labels == 0)
            hy_muons = (labels == 2)
            print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
            # basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)
            basic_cuts = (nhit_cut & towall_cut)


            # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
            stride1 = eval_output_path
            # run_result = [WatChMaLClassification(stride1, f'test with {percents[i]}% dead PMTs', labels, idx, basic_cuts, color=colors[i], linestyle='-')]
            run_result = [WatChMaLClassification(stride1, f'{percents[i]}%', labels, idx, basic_cuts, color=colors[i], linestyle='-')]

            (fpr, tpr) = compute_ROC(run_result, self.signal_label, self.background_labels)
            roc_curves_dict[percents[i]].append((fpr, tpr))
            auc = metrics.auc(fpr, tpr)
            auc_dict[percents[i]].append(auc)
            
        # roc_desc = self.roc_desc

        base_tpr = np.linspace(0, 1, 1001)
        epsilon = 1e-10
        mean_roc_curves = {}
        mean_aucs = {}
        for p in list(set(percents)):
            rejs = []
            aucs = []
            for i in range(len(roc_curves_dict[p])):
                (fpr, tpr) = roc_curves_dict[p][i]
                
                with np.errstate(divide='ignore'):
                    # rej = 1 / (fpr + epsilon)
                    rej = 1 / fpr

                # interpolate ROC rejection curve
                # rej = 1/fpr
                rej = np.interp(base_tpr, tpr, rej)
                rejs.append(rej)

                aucs.append(auc_dict[p][i])

            
            rejs = np.array(rejs)
            mean_rejs = rejs.mean(axis=0)
            mean_roc_curves[p] = mean_rejs

            aucs = np.array(aucs)
            mean_aucs[p] = np.mean(aucs)

        # set the results
        self.roc_curves_dict = roc_curves_dict
        self.auc_dict = auc_dict

        self.base_tpr = base_tpr
        self.mean_roc_curves = mean_roc_curves
        self.colors = colors
        self.mean_aucs = mean_aucs

        self.computed = True
        return

    def save_AUC_summary_stats(self):
        '''
        Requires: auc_dict has to be computed
        '''
        if self.computed is False:
            self.compute_mean_ROCs()

        auc_summary = None
        # for p in sorted_percents:
        for p in sorted(list(set(self.percents))):
            aucs_group_by_p = self.auc_dict[p]
            
            if len(aucs_group_by_p) != 0:
                x = aucs_group_by_p
                auc_s_p = np.array([
                    p,
                    len(x),
                    np.min(x),
                    np.percentile(x, 25),
                    np.percentile(x, 50),
                    np.mean(x),
                    np.percentile(x, 75),
                    np.max(x),
                    np.std(x)
                ])
                # auc_s_p = np.insert(auc_s_p, 0, p)
                print(f"auc summary for {p} percents: ", auc_s_p)
                auc_summary = np.vstack((auc_summary, auc_s_p)) if auc_summary is not None else auc_s_p
                    
        print('auc summary stats by percents!', auc_summary)
        header_str = ",".join(self.summary_stats_header)
        np.savetxt(self.settings.outputPlotPath + f'/summary_stats_{self.roc_desc}_AUCs.csv', auc_summary, header=header_str, delimiter=',')
        return auc_summary
    
    def plot_AUC_summary_stats(self):
        df = pd.DataFrame(self.save_AUC_summary_stats(), columns=self.summary_stats_header)
        print("self", self.summary_stats_header)
        # df.columns = self.summary_stats_header
        # print(df)

        df['Sample SD'] = df.apply(lambda row: row['SD'] / np.sqrt(row['Count']), axis=1)

        plt.errorbar(x = df['Dead PMT Rate (%)'], y = df['Mean'], yerr = df['Sample SD'], fmt='o')
        plt.xlabel('Dead PMT Rate [%]')
        plt.ylabel('AUC')
        auc_plot_title = self.roc_desc.replace('_', ' ')
        plt.title(f'Mean AUCs with Standard Error ({auc_plot_title})')

        plt.savefig(self.settings.outputPlotPath + f'{self.roc_desc}_auc_errorbar.png')
        plt.clf()
        

    def generate_random_color(seed=42):
        random.seed(seed)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f'#{r:02X}{g:02X}{b:02X}'
        

# def analyze_multiple_classification(settings, sub_dir_names, percents, tasks=['roc']):
#     """
#     Call functions based on `tasks` that user wants to do
    
#     Params
#     ------
#     settings:
#     sub_dir_names:
#     percents:
#     tasks: a list of str from below
#         'roc' to plot mean ROC curves grouped by dead PMT rates
#         'summary' to save summary statistics of AUC
#         'superimposed' to plot superimposed ROC curves from all evaluations (this can be messy)
#     """
    
#     for t in tasks:
#         if t == 'roc':
#             # plot_simple_ROCs(settings, sub_dir_names, percents)
#             plot_mean_ROCs(settings, sub_dir_names, percents, None, None)
            
    
#     return

# def plot_superimposed_ROC(settings, sub_dir_names, percents):
#     # fail check based on contents of the params

#     # base_path + sub_dirs[i] is where result of i-th evaluation is stored.
#     base_path = settings.mlPath
#     sub_dirs = sub_dir_names

#     # colors list. i-th color will be color of ROC curve from i-th evaluation
#     # colors = ['black', 'red','pink','orange', 'blue', 'skyblue', 'purple', 'yellow', 'green', 'brown','yellow', 'green', 'brown']
#     colors = [generate_random_color() for _ in range(len(percents))]
#     # percents list. i-th value = dead pmt percent for i-th evaluation. Used for legends in plot.
#     # if percents is None:
#     #     percents = [0, 3, 3, 3, 5, 5, 5, 100, 100, 100, 100, 100, 100, 100]
#     # percents = [0, 3, 3, 3, 5, 5, 5]
    
#     ##########
#     fig_roc = None
#     ax_roc = None
#     aucs = None
#     for i, sub_dir in enumerate(sub_dirs):
#         eval_output_path = base_path + sub_dir

#         idx = np.array(sorted(np.load(str(eval_output_path) + "/indices.npy")))
#         idx = np.unique(idx)
#         softmax = np.array(np.load(str(eval_output_path) + "/softmax.npy"))
        
#         labels_test = np.array(np.load(str(eval_output_path) + "/labels.npy"))
#         #positions_array = np.array(np.load(str(newest_directory) + "/pred_positions.npy"))
#         #true_positions_array = np.array(np.load(str(newest_directory) + "/true_positions.npy"))

#         # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
#         hy = h5py.File(settings.inputPath + "/combine_combine.hy", "r")
#         print(hy["labels"].shape)
#         print(np.amax(idx))
#         angles = np.array(hy['angles'])[idx].squeeze() 
#         labels = np.array(hy['labels'])[idx].squeeze() 
#         veto = np.array(hy['veto'])[idx].squeeze()
#         energies = np.array(hy['energies'])[idx].squeeze()
#         positions = np.array(hy['positions'])[idx].squeeze()
#         #positions=true_positions_array.squeeze()
#         directions = math.direction_from_angles(angles)
#         rootfiles = np.array(hy['root_files'])[idx].squeeze()
#         event_ids = np.array(hy['event_ids'])[idx].squeeze()
#         #positions_ml = positions_array.squeeze()

#         # calculate number of hits 
#         events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
#         nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()



#         #Save ids and rootfiles to compare to fitqun, after applying cuts
#         ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)


#         softmax_sum = np.sum(softmax,axis=1)
#         print(f"SOFTMAX SUM: {np.amin(softmax_sum)}")

#         # calculate additional parameters 
#         towall = math.towall(positions, angles, tank_axis = 2)
#         ml_cheThr = list(map(get_cherenkov_threshold, labels))

        
#         # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
#         nhit_cut = nhits > 0 #25
#         towall_cut = towall > 100
#         # veto_cut = (veto == 0)
#         hy_electrons = (labels == 0)
#         hy_muons = (labels == 2)
#         print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
#         # basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)
#         basic_cuts = (nhit_cut & towall_cut)
        

#         # set class labels and decrease values within labels to match either 0 or 1 
#         if type(settings.signalLabels) == list:
#             signal_label = [settings.signalLabels] # previously e_label
#         else:
#             signal_label = [settings.signalLabels]
#         if type(settings.bkgLabels) == list:
#             background_labels = settings.bkgLabels # previously mu_label
#         else:
#             background_labels = [settings.bkgLabels]
#         #labels = [x - 1 for x in labels]

#         # print("signal from settings:", settings.signalLabels)
#         # print("bckgrd from settings:", settings.bkgLabels)

#         # print("e_label  ", signal_label)
#         # print("mu_label ", background_labels)
        
#         label_names = ['Muon', 'Electron', 'Pion'] # The labels are muons 0, electrons 1, and pions 2
#         signal_label_desc = label_names[signal_label[0]]
#         background_labels_desc = 'Others' if len(background_labels) > 1 else label_names[background_labels[0]]


#         # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
#         stride1 = eval_output_path
#         # run_result = [WatChMaLClassification(stride1, f'test with {percents[i]}% dead PMTs', labels, idx, basic_cuts, color=colors[i], linestyle='-')]
#         run_result = [WatChMaLClassification(stride1, f'{percents[i]}%', labels, idx, basic_cuts, color=colors[i], linestyle='-')]
  
#         fig_roc, ax_roc = plot_rocs(run_result, signal_label, background_labels, ax = ax_roc, selection=basic_cuts, x_label=f"{signal_label_desc} Tagging Efficiency", y_label=f"{background_labels_desc} Rejection",
#                 legend=None, mode='rejection', fitqun=None, label='ML', fig_size =(9, 8))
#         # fig_roc, ax_roc = plot_rocs(run_result, signal_label, background_labels, ax = ax_roc, selection=basic_cuts, x_label=f"{signal_label_desc} Tagging Efficiency", y_label=f"{background_labels_desc} Rejection",
#         #         legend='best', mode='rejection', fitqun=None, label='ML', fig_size =(9, 8))
#         auc = compute_AUC(run_result, signal_label, background_labels)
        
#         ax_roc.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

#         # print('auc: ', auc)
#         auc_current = np.array([percents[i], auc])
#         if aucs is None:
#             aucs = auc_current
#         else:
#             aucs = np.vstack((aucs, auc_current))
    
#     roc_desc = f'{signal_label_desc}_vs_{background_labels_desc}'
     
#     fig_roc.savefig(settings.outputPlotPath + f'/{roc_desc}_ROCs.png', format='png')
#     np.savetxt(settings.outputPlotPath + f'/{roc_desc}_AUCs_all.csv', aucs, header="percent, AUC", delimiter=',')

#     save_summary_stats_AUC(settings=settings, aucs=aucs, roc_desc=roc_desc, percents=percents)

#     print(f'finished running plots and saving results for {roc_desc}')

#     # 1st col is percent, and 2nd col is auc
#     return aucs

# def save_summary_stats_AUC(settings, aucs, roc_desc, percents):
#     auc_summary = None
#     # for p in sorted_percents:
#     for p in list(set(percents)).sort():
#         aucs_group_by_p = aucs[aucs[:, 0] == p][:, 1]
        
#         if aucs_group_by_p.size != 0:
#             x = aucs_group_by_p
#             auc_s_p = np.array([
#                 p,
#                 np.min(x),
#                 np.percentile(x, 25),
#                 np.percentile(x, 50),
#                 np.mean(x),
#                 np.percentile(x, 75),
#                 np.max(x),
#                 np.std(x)
#             ])
#             # auc_s_p = np.insert(auc_s_p, 0, p)
#             print(f"auc summary for {p} percents: ", auc_s_p)
#             auc_summary = np.vstack((auc_summary, auc_s_p)) if auc_summary is not None else auc_s_p
                
#     print('auc summary stats by percents!', auc_summary)
#     np.savetxt(settings.outputPlotPath + f'/summary_stats_{roc_desc}_AUCs.csv', auc_summary, header="Dead PMT Rate (%), Min, Q1, Median, Mean, Q3, Max, SD", delimiter=',')

# def plot_mean_ROCs(settings, sub_dir_names, percents, xlim, ylim):
#     if global_base_tpr is None or global_mean_roc_curves is None or global_colors is None:
#         plot_simple_ROCs(settings, sub_dir_names, percents)
    
#     fig, ax = plt.subplots()
#     base_tpr = global_base_tpr
#     mean_roc_curves = global_mean_roc_curves
#     colors = global_colors
#     mean_aucs = global_mean_aucs

#     for i, p in enumerate(list(set(percents))):
#         ax.plot(base_tpr, mean_roc_curves[p], colors[i], label = f'Mean ROC({p}% Dead, AUC={round(mean_aucs[p], 4)})')
    

#     # set class labels and decrease values within labels to match either 0 or 1 
#         if type(settings.signalLabels) == list:
#             signal_label = [settings.signalLabels] # previously e_label
#         else:
#             signal_label = [settings.signalLabels]
#         if type(settings.bkgLabels) == list:
#             background_labels = settings.bkgLabels # previously mu_label
#         else:
#             background_labels = [settings.bkgLabels]
        
#         label_names = ['Muon', 'Electron', 'Pion'] # The labels are muons 0, electrons 1, and pions 2
#         signal_label_desc = label_names[signal_label[0]]
#         background_labels_desc = 'Others' if len(background_labels) > 1 else label_names[background_labels[0]]
    
#     roc_desc = f'{signal_label_desc}_vs_{background_labels_desc}'
    
#     ax.set_yscale('log')
#     ax.set_xlabel(f'{signal_label_desc} Tagging Efficiency')
#     ax.set_ylabel(f'{background_labels_desc} Rejection')
#     roc_title = roc_desc.replace('_', ' ')
#     ax.set_title(f'{roc_title} Mean ROCs by Dead PMT Rates (%)')
#     ax.legend()
    
#     if xlim is not None:
#         ax.set_xlim(xlim)
#     if ylim is not None:
#         ax.set_ylim(ylim)
#     fig.savefig(settings.outputPlotPath + f'/{roc_desc}_ROCs_mean_all_auc_{g_count}.png', format='png')
#     g_count += 1


# def plot_simple_ROCs(settings, sub_dir_names, percents, xlim=None, ylim=None):
#     # fail check based on contents of the params

#     # base_path + sub_dirs[i] is where result of i-th evaluation is stored.
#     base_path = settings.mlPath
#     sub_dirs = sub_dir_names

#     colors = [generate_random_color(p) for p in range(len(percents))]
    
    
#     ##########
    
#     roc_curves_dict = {}
#     auc_dict = {}
#     for p in list(set(percents)):
#         roc_curves_dict[p] = []
#         auc_dict[p] = []

#     for i, sub_dir in enumerate(sub_dirs):
#         roc_curves = []

#         eval_output_path = base_path + sub_dir

#         idx = np.array(sorted(np.load(str(eval_output_path) + "/indices.npy")))
#         idx = np.unique(idx)
#         softmax = np.array(np.load(str(eval_output_path) + "/softmax.npy"))
        
#         labels_test = np.array(np.load(str(eval_output_path) + "/labels.npy"))

#         # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
#         hy = h5py.File(settings.inputPath + "/combine_combine.hy", "r")
#         print(hy["labels"].shape)
#         print(np.amax(idx))
#         angles = np.array(hy['angles'])[idx].squeeze() 
#         labels = np.array(hy['labels'])[idx].squeeze() 
#         veto = np.array(hy['veto'])[idx].squeeze()
#         energies = np.array(hy['energies'])[idx].squeeze()
#         positions = np.array(hy['positions'])[idx].squeeze()
#         #positions=true_positions_array.squeeze()
#         directions = math.direction_from_angles(angles)
#         rootfiles = np.array(hy['root_files'])[idx].squeeze()
#         event_ids = np.array(hy['event_ids'])[idx].squeeze()
#         #positions_ml = positions_array.squeeze()

#         # calculate number of hits 
#         events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
#         nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()

#         #Save ids and rootfiles to compare to fitqun, after applying cuts
#         ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)

#         softmax_sum = np.sum(softmax,axis=1)
#         print(f"SOFTMAX SUM: {np.amin(softmax_sum)}")

#         # calculate additional parameters 
#         towall = math.towall(positions, angles, tank_axis = 2)
#         ml_cheThr = list(map(get_cherenkov_threshold, labels))

#         # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
#         nhit_cut = nhits > 0 #25
#         towall_cut = towall > 100
#         # veto_cut = (veto == 0)
#         hy_electrons = (labels == 0)
#         hy_muons = (labels == 2)
#         print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
#         # basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)
#         basic_cuts = (nhit_cut & towall_cut)

#         # set class labels and decrease values within labels to match either 0 or 1 
#         if type(settings.signalLabels) == list:
#             signal_label = [settings.signalLabels] # previously e_label
#         else:
#             signal_label = [settings.signalLabels]
#         if type(settings.bkgLabels) == list:
#             background_labels = settings.bkgLabels # previously mu_label
#         else:
#             background_labels = [settings.bkgLabels]
        
#         label_names = ['Muon', 'Electron', 'Pion'] # The labels are muons 0, electrons 1, and pions 2
#         signal_label_desc = label_names[signal_label[0]]
#         background_labels_desc = 'Others' if len(background_labels) > 1 else label_names[background_labels[0]]


#         # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
#         stride1 = eval_output_path
#         # run_result = [WatChMaLClassification(stride1, f'test with {percents[i]}% dead PMTs', labels, idx, basic_cuts, color=colors[i], linestyle='-')]
#         run_result = [WatChMaLClassification(stride1, f'{percents[i]}%', labels, idx, basic_cuts, color=colors[i], linestyle='-')]

#         (fpr, tpr) = compute_ROC(run_result, signal_label, background_labels)
#         roc_curves_dict[percents[i]].append((fpr, tpr))
#         auc = metrics.auc(fpr, tpr)
#         auc_dict[percents[i]].append(auc)
        
    
#     roc_desc = f'{signal_label_desc}_vs_{background_labels_desc}'
     
#     # fig_roc.savefig(settings.outputPlotPath + f'/{roc_desc}_ROCs.png', format='png')
#     # np.savetxt(settings.outputPlotPath + f'/{roc_desc}_AUCs_all.csv', aucs, header="percent, AUC", delimiter=',')

#     # save_summary_stats_AUC(settings=settings, aucs=aucs, roc_desc=roc_desc, percents=percents)

#     print(f'finished running plots and saving results for {roc_desc}')

#     base_tpr = np.linspace(0, 1, 1001)
#     epsilon = 1e-10
#     mean_roc_curves = {}
#     mean_aucs = {}
#     for p in list(set(percents)):
#         rejs = []
#         aucs = []
#         for i in range(len(roc_curves_dict[p])):
#             (fpr, tpr) = roc_curves_dict[p][i]
            
#             with np.errstate(divide='ignore'):
#                 # rej = 1 / (fpr + epsilon)
#                 rej = 1 / fpr

#             # interpolate ROC rejection curve
#             # rej = 1/fpr
#             rej = np.interp(base_tpr, tpr, rej)
#             rejs.append(rej)

#             aucs.append(auc_dict[p][i])

        
#         rejs = np.array(rejs)
#         mean_rejs = rejs.mean(axis=0)
#         mean_roc_curves[p] = mean_rejs

#         aucs = np.array(aucs)
#         mean_aucs[p] = np.mean(aucs)
    
    

    

#     fig, ax = plt.subplots()
#     # fig.set_size_inches(10, 6)

#     colors = ['blue', 'g', 'r', 'violet', 'k', 'c', 'm', 'orange', 'purple', 'brown']
#     for i, p in enumerate(list(set(percents))):
#         ax.plot(base_tpr, mean_roc_curves[p], colors[i], label = f'Mean ROC({p}% Dead, AUC={round(mean_aucs[p], 4)})')
    
#     ax.set_yscale('log')
#     ax.set_xlabel(f'{signal_label_desc} Tagging Efficiency')
#     ax.set_ylabel(f'{background_labels_desc} Rejection')
#     roc_title = roc_desc.replace('_', ' ')
#     ax.set_title(f'{roc_title} Mean ROCs by Dead PMT Rates (%)')
#     ax.legend()
    
#     ax.set_xlim([0.4, 1.03])
#     ax.set_ylim([0.1, 110])

#     # fig.savefig(settings.outputPlotPath + f'/{roc_desc}_ROCs_mean_all_auc_zoom.png', format='png')

#     global_base_tpr = base_tpr
#     global_mean_roc_curves = mean_roc_curves
#     global_colors = colors
#     global_mean_aucs = mean_aucs
#     return

# def some_exp(settings, sub_dir_names=None, percents=None):
#     base_path = '/data/thoriba/t2k/eval/oct20_eMuPosPion_0dwallCut_flat_1/09052024-171021/'
#     sub_dirs = sub_dir_names
#     if sub_dirs is None:
#         sub_dirs = [
#             'multiEval_seed_0_0th_itr_0_percent_20240530101357',

#             'multiEval_seed_0_0th_itr_3_percent_20240529134944',
#             'multiEval_seed_1_1th_itr_3_percent_20240529140157',
#             'multiEval_seed_2_2th_itr_3_percent_20240529141406',

#             'multiEval_seed_0_0th_itr_5_percent_20240529142605',
#             'multiEval_seed_1_1th_itr_5_percent_20240529143807',
#             'multiEval_seed_2_2th_itr_5_percent_20240529145014'
#         ]
#     # colors list. i-th color will be color of ROC curve from i-th evaluation
#     colors = ['black', 'red','pink','orange', 'blue', 'skyblue', 'purple', 'yellow', 'green', 'brown','yellow', 'green', 'brown']
#     # percents list. i-th value = dead pmt percent for i-th evaluation. Used for legends in plot.
#     if percents is None:
#         percents = [0, 3, 3, 3, 5, 5, 5, 100, 100, 100, 100, 100, 100, 100]
    
#     ##########
#     fig_roc = None
#     ax_roc = None
#     aucs = None
#     for i, sub_dir in enumerate(sub_dirs):
#         settings.mlPath = base_path + sub_dir


#         idx = np.array(sorted(np.load(str(settings.mlPath) + "/indices.npy")))
#         idx = np.unique(idx)
#         softmax = np.array(np.load(str(settings.mlPath) + "/softmax.npy"))
        
#         labels_test = np.array(np.load(str(settings.mlPath) + "/labels.npy"))
#         #positions_array = np.array(np.load(str(newest_directory) + "/pred_positions.npy"))
#         #true_positions_array = np.array(np.load(str(newest_directory) + "/true_positions.npy"))

#         # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
#         hy = h5py.File(settings.inputPath + "/combine_combine.hy", "r")
#         print(hy["labels"].shape)
#         print(np.amax(idx))
#         angles = np.array(hy['angles'])[idx].squeeze() 
#         labels = np.array(hy['labels'])[idx].squeeze() 
#         veto = np.array(hy['veto'])[idx].squeeze()
#         energies = np.array(hy['energies'])[idx].squeeze()
#         positions = np.array(hy['positions'])[idx].squeeze()
#         #positions=true_positions_array.squeeze()
#         directions = math.direction_from_angles(angles)
#         rootfiles = np.array(hy['root_files'])[idx].squeeze()
#         event_ids = np.array(hy['event_ids'])[idx].squeeze()
#         #positions_ml = positions_array.squeeze()

#         # calculate number of hits 
#         events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
#         nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()



#         #Save ids and rootfiles to compare to fitqun, after applying cuts
#         ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)


#         softmax_sum = np.sum(softmax,axis=1)
#         print(f"SOFTMAX SUM: {np.amin(softmax_sum)}")

#         # calculate additional parameters 
#         towall = math.towall(positions, angles, tank_axis = 2)
#         ml_cheThr = list(map(get_cherenkov_threshold, labels))

        
#         # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
#         nhit_cut = nhits > 0 #25
#         towall_cut = towall > 100
#         # veto_cut = (veto == 0)
#         hy_electrons = (labels == 0)
#         hy_muons = (labels == 2)
#         print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
#         # basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)
#         basic_cuts = (nhit_cut & towall_cut)
        

#         # set class labels and decrease values within labels to match either 0 or 1 
#         if type(settings.signalLabels) == list:
#             signal_label = [settings.signalLabels] # previously e_label
#         else:
#             signal_label = [settings.signalLabels]
#         if type(settings.bkgLabels) == list:
#             background_labels = settings.bkgLabels # previously mu_label
#         else:
#             background_labels = [settings.bkgLabels]
#         #labels = [x - 1 for x in labels]

#         print("signal from settings:", settings.signalLabels)
#         print("bckgrd from settings:", settings.bkgLabels)

#         print("e_label  ", signal_label)
#         print("mu_label ", background_labels)
        
#         label_names = ['Muon', 'Electron', 'Pion'] # The labels are muons 0, electrons 1, and pions 2
#         signal_label_desc = label_names[signal_label[0]]
#         background_labels_desc = 'Others' if len(background_labels) > 1 else label_names[background_labels[0]]


#         # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
#         stride1 = settings.mlPath
#         run_result = [WatChMaLClassification(stride1, f'test with {percents[i]}% dead PMTs', labels, idx, basic_cuts, color=colors[i], linestyle='-')]
#         # print(f"UNIQUE IN LABLES: {np.unique(fitqun_labels, return_counts=True)}")

        
#         # for single runs and then can plot the ROC curves with it 
#         #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

#         plot_rocs(run_result, signal_label, background_labels, ax = ax_roc, selection=basic_cuts, x_label=f"{signal_label_desc} Tagging Efficiency", y_label=f"{background_labels_desc} Rejection",
#                 legend='best', mode='rejection', fitqun=None, label='ML', fig_size =(9, 8))
        
        
#         print('auc from compute_AUC: ', compute_AUC(run_result, signal_label, background_labels))
        
#         # print('auc: ', auc)
#         # if aucs is None:
#         #     aucs = np.array([auc])
#         # else:
#         #     aucs = np.vstack(aucs, np.array[auc])
     
#     # fig_roc.savefig(settings.outputPlotPath + f'/{signal_label_desc}_vs_{background_labels_desc}_ROCs.png', format='png')
#     # np.savetxt(settings.outputPlotPath + f'/{signal_label_desc}_vs_{background_labels_desc}_ROCs.csv', aucs)

#     # remove comment for ROC curves of single run 