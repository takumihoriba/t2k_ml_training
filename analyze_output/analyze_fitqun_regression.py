import numpy as np

import os

import h5py

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


from scipy.optimize import curve_fit
from scipy.stats.mstats import mquantiles_cimj

import WatChMaL.analysis.utils.fitqun as fq

from tqdm import tqdm

import analysis.utils.math as math

from plotting import regression_analysis, regression_analysis_perVar
from analyze_output.utils.math import get_cherenkov_threshold



def analyze_fitqun_regression(settings):
     '''
     Plot fiTQun specific regression results using un_normalize(), regression_analysis(), and read_fitqun_file().
     Args:
         hy_path (str, optional): directory where fitqun_combine.hy and combine_combine.hy files are located. 
         true_path (str, optional): directory where true_positions.npy file is located.
     Returns:
         None
     '''
     # get values out of fitqun file, where mu_1rpos and e_1rpos are the positions of muons and electrons respectively
     (_, labels, _, fitqun_hash), (mu_1rpos, e_1rpos, mu_1rdir, e_1rdir, mu_1rmom, e_1rmom) = fq.read_fitqun_file(settings.inputPath+'/fitqun_combine.hy', regression=True)

     # read in the indices file
     idx = np.array(sorted(np.load(settings.mlPath + "/indices.npy")))

     # read in the main HDF5 file that has the rootfiles and event_ids
     hy = h5py.File(settings.inputPath+'/combine_combine.hy', "r")
     positions = np.array(hy['positions'])[idx].squeeze()
     directions = np.array(hy['directions'])[idx].squeeze()
     energies = np.array(hy['energies'])[idx].squeeze()
     labels = np.array(hy['labels'])[idx].squeeze()
     momenta = np.ones(energies.shape[0])
     #momenta[labels == 1] = np.sqrt(np.multiply(energies[labels==1], energies[labels==1]) - np.multiply(momenta[labels==1]*0.5,momenta[labels==1]*0.5))
     momenta = np.sqrt(np.multiply(energies, energies) - np.multiply(momenta*0.5,momenta*0.5))
     #momenta[labels == 0] = np.sqrt(np.multiply(energies[labels==0], energies[labels==0]) - np.multiply(momenta[labels==0]*105.7,momenta[labels==0]*105.7))
     #momenta[labels == 2] = np.sqrt(np.multiply(energies[labels==2], energies[labels==2]) - np.multiply(momenta[labels==2]*139.584,momenta[labels==2]*139.584))
     rootfiles = np.array(hy['root_files'])[idx].squeeze()
     event_ids = np.array(hy['event_ids'])[idx].squeeze()
     angles = math.angles_from_direction(directions)
     towall = math.towall(positions, angles, tank_axis = 2)


     # calculate number of hits 
     events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
     nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()
     nhits_cut=200


     #Apply cuts
     event_ids = event_ids[(nhits> nhits_cut)]
     rootfiles = rootfiles[(nhits> nhits_cut)]
     positions = positions[(nhits> nhits_cut)]
     directions = directions[(nhits> nhits_cut)]
     labels = labels[(nhits> nhits_cut)]
     momenta = momenta[(nhits> nhits_cut)]
     energies = energies[(nhits> nhits_cut)]
     towall = towall[(nhits> nhits_cut)]
     nhits = nhits[(nhits> nhits_cut)]

     ml_hash = fq.get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)


     # load in the true positions 
     # Not necessary in current config
     '''
     true_target = np.load(npy_path + target+".npy")
     print('true_'+target+'.shape =', true_target.shape)

     # unnormalize the true_positions
     if 'positions' in target:
        tp = []
        for t in true_target:
            tp.append(un_normalize(t))
        true_target = np.array(tp)

     print('true_'+target+' =', true_target)
     print('mu_1rpos =', mu_1rpos)
     print('e_1rpos =', e_1rpos)
     '''

     # use intersect1d to find the intersection of fitqun_hash and ml_hash, specifically, intersect is a 
     # sorted 1D array of common and unique elements, comm1 is the indices of the first occurrences of 
     # the common values in fitqun_hash, comm2 is the indices of the first occurrences of the common values 
     # in ml_hash. So we use comm1 to index fitqun_hash and comm2 to index ml_hash.
     intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, return_indices=True)
     fitqun_labels = labels[comm2]
     fitqun_mu_1rpos = mu_1rpos[comm1].squeeze() 
     fitqun_e_1rpos = e_1rpos[comm1].squeeze() 
     fitqun_mu_1rdir = mu_1rdir[comm1].squeeze() 
     fitqun_e_1rdir = e_1rdir[comm1].squeeze() 
     fitqun_mu_1rmom = mu_1rmom[comm1].squeeze() 
     fitqun_e_1rmom = e_1rmom[comm1].squeeze() 
     positions = positions[comm2]
     directions = directions[comm2]
     momenta = momenta[comm2]
     energies = energies[comm2]
     towall = towall[comm2]
     nhits = nhits[comm2]
     cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
     visible_energy = energies - cheThr

     if "positions" in settings.target:
         truth = positions
         fitqun_mu = fitqun_mu_1rpos
         fitqun_e = fitqun_e_1rpos
     elif "directions" in settings.target:
        truth = directions
        fitqun_mu = fitqun_mu_1rdir
        fitqun_e = fitqun_e_1rdir
     elif "momentum" in settings.target or "momenta" in settings.target:
        truth = momenta
        fitqun_mu = fitqun_mu_1rmom
        fitqun_e = fitqun_e_1rmom

     true_0, pred_0, ve_0, tw_0, nhits_0, dir_0 = [], [], [], [], [], []
     true_1, pred_1, ve_1, tw_1, nhits_1, dir_1 = [], [], [], [], [], []
     true_2, pred_2, ve_2, tw_2, nhits_2, dir_2 = [], [], [], [], [], []
     
     for i in range(len(fitqun_labels)):
         # LABEL 0 - muons  
         if fitqun_labels[i] == 0:
             true_0.append(truth[i])
             pred_0.append(fitqun_mu[i])
             ve_0.append(visible_energy[i])
             tw_0.append(towall[i])
             nhits_0.append(nhits[i])
             dir_0.append(directions[i])

         # LABEL 1 - electrons  
         elif fitqun_labels[i] == 1:
             true_1.append(truth[i])
             pred_1.append(fitqun_e[i])
             ve_1.append(visible_energy[i])
             tw_1.append(towall[i])
             nhits_1.append(nhits[i])
             dir_1.append(directions[i])
        
        # label 2 -- pions
         else:
             true_2.append(truth[i])
             pred_2.append(fitqun_e[i])
             ve_2.append(visible_energy[i])
             tw_2.append(towall[i])
             nhits_2.append(nhits[i])
             dir_2.append(directions[i])

     # convert lists to arrayss
     true_0 = np.array(true_0)
     true_1 = np.array(true_1)
     pred_0 = np.array(pred_0)
     pred_1 = np.array(pred_1)
     tw_0 = np.array(tw_0)
     tw_1 = np.array(tw_1)
     dir_0 = np.array(dir_0)
     dir_1 = np.array(dir_1)

     #print(true_0.shape)
     single_analysis = []
     multi_analysis = {}
     if settings.particleLabel==0:
        print('######## fiTQun MUON EVENTS ########')
        vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target=settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
        single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=tw_0)
        multi_analysis['towall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_0, pred=pred_0, dir=dir_0, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=ve_0)
        multi_analysis['ve'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     if settings.particleLabel==1:
        print('######## fiTQun ELECTRON EVENTS ########')
        #regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, target = target, extra_string="fitqun_Electrons", save_plots=False, variable=ve_1)
        vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst = regression_analysis(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target=settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=True, plot_path = settings.outputPlotPath)
        single_analysis = [vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst] 
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=tw_1)
        multi_analysis['towall'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]
        bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict = regression_analysis_perVar(from_path=False, true=true_1, pred=pred_1, dir=dir_1, target = settings.target, extra_string="fiTQun_"+settings.plotName, save_plots=False, variable=ve_1)
        multi_analysis['ve'] = [bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict]

     return single_analysis, multi_analysis
