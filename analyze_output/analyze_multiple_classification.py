import numpy as np

from analyze_output.utils.plotting import softmax_plots
from analyze_output.utils.math import get_cherenkov_threshold

import os
import random
import pandas as pd

import h5py

# from analysis.classification import plot_rocs2

from analysis.classification import WatChMaLClassification
# from analysis.classification import plot_efficiency_profile, plot_rocs
from analysis.classification import compute_AUC, compute_ROC

# from analysis.utils.plotting import plot_legend
# from analysis.utils.binning import get_binning
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


class MultiClassificationAnalysis:
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
    
    def set_colors(self):
        colors = []
        for s in range(len(self.percents)):
            self.generate_random_color(s)

    
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

        print('This instance of MultiClassificationAnalysis works on ', self.roc_desc)
    
    def analyze(self, tasks=['roc']):
        '''
        Analyzes the performance of several evaluations of model
        tasks: list of strings
            'all': do everything
            'roc': 
            'auc':
            'rej':
        '''
        if tasks[0] == 'all':
            self.plot_mean_ROCs()
            self.plot_mean_ROCs(xlim=None, ylim=[100, None])
            self.plot_mean_ROCs(xlim=[.8, None], ylim=[100, None])
            
            self.plot_AUC_summary_stats()
            self.plot_rejections_summary_stats(0.9)
            self.plot_rejections_summary_stats(0.95)
            self.plot_rejections_summary_stats(0.99)
            return
            
        for t in tasks:
            if t == 'roc':
                self.plot_mean_ROCs()
                self.plot_mean_ROCs(xlim=None, ylim=[100, 100000])
            elif t == 'rej':
                self.plot_rejections_summary_stats(0.9)
                self.plot_rejections_summary_stats(0.95)
                self.plot_rejections_summary_stats(0.99)
            elif t == 'auc':
                self.plot_AUC_summary_stats()
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
        
        ax.grid(True, linestyle='--', linewidth=0.5)

        fig.savefig(self.settings.outputPlotPath + f'/{self.roc_desc}_meanROCs_{self.plot_counter}.png', format='png')
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
            print('----------------------------------')
            print(f'Processing {i}th (of {len(sub_dirs)}) evaluation')
            roc_curves = []

            eval_output_path = base_path + sub_dir

            idx = np.array(sorted(np.load(str(eval_output_path) + "/indices.npy")))
            idx = np.unique(idx)
            softmax = np.array(np.load(str(eval_output_path) + "/softmax.npy"))
            
            labels_test = np.array(np.load(str(eval_output_path) + "/labels.npy"))

            # here I can be more efficient by moving this outside the loop since we expect that hy is same for all evaluation
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


            # 0, 1, 2 corresponds 'Muon', 'Electron', 'Pion'
            # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
            nhit_cut = nhits > 0 #25
            towall_cut = towall > 100
            # veto_cut = (veto == 0)
            # hy_electrons = (labels == 0)
            # hy_muons = (labels == 2)
            # print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
            # basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)
            # basic_cuts = (nhit_cut & towall_cut)

            signal_cut = np.isin(labels, self.signal_label)
            bckgrd_cut = np.isin(labels, self.background_labels)

            basic_cuts = (signal_cut | bckgrd_cut) & nhit_cut & towall_cut

            print(f'Signal = {self.signal_label_desc}; Background = {self.background_labels_desc}')
            print(f"""signal_cut (previously hy_electrons): {signal_cut.shape}, bckgrd_cut (previsouly hy_muons): {bckgrd_cut.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}""")

            # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
            stride1 = eval_output_path
            # run_result = [WatChMaLClassification(stride1, f'test with {percents[i]}% dead PMTs', labels, idx, basic_cuts, color=colors[i], linestyle='-')]
            run_result = [WatChMaLClassification(stride1, f'{percents[i]}%', labels, idx, basic_cuts, color='b', linestyle='-')]

            (fpr, tpr) = compute_ROC(run_result, self.signal_label, self.background_labels, basic_cuts)
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
        np.savetxt(self.settings.outputPlotPath + f'/{self.roc_desc}_auc_summary.csv', auc_summary, header=header_str, delimiter=',')
        return auc_summary
    
    def plot_AUC_summary_stats(self):
        df = pd.DataFrame(self.save_AUC_summary_stats(), columns=self.summary_stats_header)
        df['Sample SD'] = df.apply(lambda row: row['SD'] / np.sqrt(row['Count']), axis=1)

        fig, ax = plt.subplots()
        
        ax.errorbar(x = df['Dead PMT Rate (%)'], y = df['Mean'], yerr = df['Sample SD'], fmt='o', markersize=5, capsize=5)
        ax.set_xlabel('Dead PMT Rate [%]')
        ax.set_ylabel('AUC')
        auc_plot_title = self.roc_desc.replace('_', ' ')
        ax.set_title(f'Mean AUCs with Standard Error ({auc_plot_title})')

        ax.grid(True, linestyle='--', linewidth=0.5)

        fig.savefig(self.settings.outputPlotPath + f'{self.roc_desc}_auc_summary_plot_{self.plot_counter}.png')
        self.plot_counter += 1
        
    # def get_rejection_for_efficiency(self, eff, roc_curve):
    #     '''
    #     Returns the rejection for a given efficiency
    #     eff: float
    #         the efficiency
    #     roc_curve: tuple of np arrays
    #         the ROC curve
    #     '''
    #     fpr, tpr = roc_curve
    #     with np.errstate(divide='ignore'):
    #         return 1 / fpr[np.argmin(np.abs(tpr - eff))]
    
    # def get_rejections(self, eff=.9):
    #     if self.computed == False:
    #         self.compute_mean_ROCs()
        
    #     rejections_by_rate = {}
        
    #     for p in list(set(self.percents)):
    #         rejs = []
    #         for i in range(len(self.roc_curves_dict[p])):
    #             fpr, tpr = self.roc_curves_dict[p][i]
    #             with np.errstate(divide='ignore'):
    #                 rej = 1 / fpr[np.argmin(np.abs(tpr - eff))]
    #             rejs.append(rej)

    #         rejections_by_rate[p] = rejs
        
    #     print("rejection_by_rate", rejections_by_rate)
        
    #     return rejections_by_rate

    # def plot_rejections_summary_stats(self, eff=0.9, file=None):
    #     dict = self.get_rejections(eff)
    #     df = pd.DataFrame([(k, v) for k, vals in dict.items() for v in vals], columns=['rate', 'rejection'])

    #     mean = df.groupby('rate').mean()
    #     sem = df.groupby('rate').sem(ddof=0)


    #     # df = pd.DataFrame(self.get_rejections(eff))
    #     fig, ax = plt.subplots()
    #     # df.columns = df.columns.astype(float)
    #     # df = df.reindex(sorted(df.columns), axis=1)
    #     # mean = df.mean()
    #     # # divided by sqrt(n) instead of sqrt(n-1)
    #     # sem = df.sem(ddof=0)
    #     # ax.errorbar(mean.index, mean, yerr=sem, fmt='o')
    #     ax.errorbar(mean.index, mean['rejection'], yerr=sem['rejection'], fmt='o')
    #     ax.set_xlabel('Dead PMT Rate (%)')
    #     ax.set_ylabel('Mean Rejection (1 / FPR)')
    #     ax.set_title(f'Mean Rejection with Standard Error at {round(eff*100)}% Efficiency by Dead PMT Rate')
    #     ax.grid(True, linestyle='--', linewidth=0.5)
    #     fig.savefig(self.settings.outputPlotPath + f'/{self.roc_desc}_rejection_at_{round(eff*100)}%_{self.plot_counter}.png', format='png')
    #     self.plot_counter += 1
    

    # def get_efficiencies(self, rejc=1000):
    #     if self.computed == False:
    #         self.compute_mean_ROCs()
        
    #     efficiencies_by_rate = {}
        
    #     for p in list(set(self.percents)):
    #         effs = []
    #         for i in range(len(self.roc_curves_dict[p])):
    #             fpr, tpr = self.roc_curves_dict[p][i]
    #             with np.errstate(divide='ignore'):
    #                 # find closest tpr to rejection rate
    #                 eff = tpr[np.argmin(np.abs(1/fpr- rejc))]
    #             effs.append(eff)

    #         efficiencies_by_rate[p] = effs
        
    #     print("efficiency_by_rate", efficiencies_by_rate)

    #     df = pd.DataFrame([(k, v) for k, vals in efficiencies_by_rate.items() for v in vals], columns=['rate', 'eff'])
    #     df.to_csv(self.settings.outputPlotPath + f'/{self.roc_desc}_eff_at_{round(rejc)}.csv')
        
    #     return efficiencies_by_rate

    # def plot_efficiencies_summary_stats(self, eff=1000, file=None):
    #     dict = self.get_efficiencies(eff)
    #     df = pd.DataFrame([(k, v) for k, vals in dict.items() for v in vals], columns=['rate', 'rejection'])

    #     mean = df.groupby('rate').mean()
    #     sem = df.groupby('rate').sem(ddof=0)


    #     # df = pd.DataFrame(self.get_rejections(eff))
    #     fig, ax = plt.subplots()
    #     # df.columns = df.columns.astype(float)
    #     # df = df.reindex(sorted(df.columns), axis=1)
    #     # mean = df.mean()
    #     # # divided by sqrt(n) instead of sqrt(n-1)
    #     # sem = df.sem(ddof=0)
    #     # ax.errorbar(mean.index, mean, yerr=sem, fmt='o')
    #     ax.errorbar(mean.index, mean['rejection'], yerr=sem['rejection'], fmt='o')
    #     ax.set_xlabel('Dead PMT Rate (%)')
    #     ax.set_ylabel('Mean Efficiency ')
    #     ax.set_title(f'Mean Efficiency with Standard Error when Rejection = {eff}')
    #     ax.grid(True, linestyle='--', linewidth=0.5)
    #     fig.savefig(self.settings.outputPlotPath + f'/{self.roc_desc}_eff_at_{round(eff)}%_{self.plot_counter}.png', format='png')
    #     self.plot_counter += 1

    def get_slices(self, intercept=1000, get_eff_from_rej=True):
        '''
        when get_eff_from_rej=True, intercept is rejection. Reasonable rejection is 100 or 1000 or more
        when get_eff_from_rej=True, intercept is efficiency. Reasonable intercept value is .9, .95 or more
        '''
        if self.computed == False:
            self.compute_mean_ROCs()

        # if get_eff_from_rej==True (== 1), then we return efficiency values
        # if get_eff_from_rej==False (== 0), we are returning rejection values
        return_value_names = ['rejection', 'efficiency'] 
        
        slices_by_rate = {}
        
        for p in list(set(self.percents)):
            slices = []
            for i in range(len(self.roc_curves_dict[p])):
                fpr, tpr = self.roc_curves_dict[p][i]
                if get_eff_from_rej:
                    with np.errstate(divide='ignore'):
                        # find closest tpr to rejection rate
                        eff = tpr[np.argmin(np.abs(1/fpr- intercept))]
                    slices.append(eff)
                else:
                    with np.errstate(divide='ignore'):
                        rej = 1 / fpr[np.argmin(np.abs(tpr - intercept))]
                    slices.append(rej)

            slices_by_rate[p] = slices

        
        print(f"{return_value_names[get_eff_from_rej]}_by_rate", slices_by_rate)

        intercept_fmt = str(round(intercept*100)) + '%' if not get_eff_from_rej else str(intercept)

        df = pd.DataFrame([(k, v) for k, vals in slices_by_rate.items() for v in vals], columns=['rate', 'eff'])
        df.to_csv(self.settings.outputPlotPath + f'/{self.roc_desc}_{return_value_names[get_eff_from_rej]}_when_{return_value_names[not get_eff_from_rej]}_is_{intercept_fmt}.csv')
        
        return slices_by_rate

    def plot_slice_ROC(self, intercept=1000, get_eff_from_rej=True, file=None):

        dict = self.get_slices(intercept, get_eff_from_rej)

        to_value_name = 'efficiency' if get_eff_from_rej else 'rejection'
        from_value_name = 'efficiency' if not get_eff_from_rej else 'rejection'

        df = pd.DataFrame([(k, v) for k, vals in dict.items() for v in vals], columns=['rate', to_value_name])

        mean = df.groupby('rate').mean()
        sem = df.groupby('rate').sem(ddof=0)

        fig, ax = plt.subplots()

        ax.errorbar(mean.index, mean[to_value_name], yerr=sem[to_value_name], fmt='o')
        ax.set_xlabel('Dead PMT Rate (%)')
        ax.set_ylabel(f"Mean {to_value_name[0].capitalize() + to_value_name[1:]}")
        intercept_fmt = str(round(intercept*100)) + '%' if not get_eff_from_rej else str(intercept)
        if get_eff_from_rej:
            ax.set_title(f'Mean Efficiency with Standard Error when Rejection = {intercept_fmt}')
        else:
            ax.set_title(f'Mean Rejection with Standard Error at {intercept_fmt} Efficiency')
        ax.grid(True, linestyle='--', linewidth=0.5)
        fig.savefig(self.settings.outputPlotPath + f'/{self.roc_desc}_{to_value_name}_when_{from_value_name}_is_{intercept_fmt}_{self.plot_counter}.png', format='png')
        self.plot_counter += 1

    def plot_slice_ROC_combined(self, intercepts=[10000, 5000, 1000], get_eff_from_rej=True, file=None):
        # set things up
        to_value_name = 'efficiency' if get_eff_from_rej else 'rejection'
        from_value_name = 'efficiency' if not get_eff_from_rej else 'rejection'
        

        fig, ax = plt.subplots()

        for intercept in intercepts:
            dict = self.get_slices(intercept, get_eff_from_rej)
            df = pd.DataFrame([(k, v) for k, vals in dict.items() for v in vals], columns=['rate', to_value_name])
            mean = df.groupby('rate').mean()
            sem = df.groupby('rate').sem(ddof=0)
            intercept_fmt = str(round(intercept*100)) + '%' if not get_eff_from_rej else str(intercept)
            ax.errorbar(mean.index, mean[to_value_name], yerr=sem[to_value_name], fmt='o', 
                        label=f'{from_value_name[0].capitalize()+from_value_name[1:]} = {intercept_fmt}',
                        capsize=4, markersize=4)

        ax.set_xlabel('Dead PMT Rate [%]')
        ax.set_ylabel(f"Mean {to_value_name[0].capitalize() + to_value_name[1:]}")
        ax.legend()
        roc_desc_fmt = self.roc_desc.replace('_', ' ')
        if get_eff_from_rej:
            ax.set_title(f'{roc_desc_fmt} Mean Efficiency with SE at Different Rejection Values')
        else:
            ax.set_title(f'{roc_desc_fmt} Mean Rejection with SE at Different Efficiency Values')
        ax.grid(True, linestyle='--', linewidth=0.5)
        fig.savefig(self.settings.outputPlotPath + f'/{self.roc_desc}_{to_value_name}_at_diff_{from_value_name}_{self.plot_counter}.png', format='png')
        self.plot_counter += 1

    def generate_random_color(self, seed=42):
        random.seed(seed)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f'#{r:02X}{g:02X}{b:02X}'
    
