import numpy as np

import os

import h5py

from WatChMaL.analysis.classification import WatChMaLClassification
from WatChMaL.analysis.classification import plot_efficiency_profile, plot_rocs
from WatChMaL.analysis.utils.plotting import plot_legend
from WatChMaL.analysis.utils.binning import get_binning
from WatChMaL.analysis.utils.fitqun import read_fitqun_file, make_fitqunlike_discr, get_rootfile_eventid_hash, plot_fitqun_comparison
import WatChMaL.analysis.utils.math as math

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


from scipy.optimize import curve_fit
from scipy.stats.mstats import mquantiles_cimj

import WatChMaL.analysis.utils.fitqun as fq

from tqdm import tqdm

from math import log10, floor, ceil

def round_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def un_normalize(data, norm_min=-1, x_bounds=(-1600,1600), y_bounds=(-1600,1600), z_bounds=(-1600,1600)):
    '''
    Assume data has been normalized to a range of [-1, 1] or [0, 1]
    this function brings the values back to their original values 

    Args:
        data (array): 1D array of data to be unnormalized.
        norm_min (int, optional): minimum of normalized data. can only be -1 or 0. 
        x_bounds (tuple, optional): real minimum and maximum value of original data on x-axis.s
        y_bounds (tuple, optional): real minimum and maximum value of original data on y-axis.
        z_bounds (tuple, optional): real minimum and maximum value of original data on z-axis.

    Returns:
        data (np.array): unnormalized data.
    '''
    bounds = [x_bounds, y_bounds, z_bounds]
    for i in range(3):

        if norm_min == 0:
            data[i] = ((data[i])*(bounds[i][1] - bounds[i][0])) + bounds[i][0] 

        elif norm_min == -1: 
            data[i] = data[i]*(bounds[i][1])
        
    return data




def regression_analysis_perVar(from_path=True, dirpath='outputs', combine=True, true=None, pred=None, dir = None, variable = None, target = 'positions', extra_string = "", save_plots=False):

    var_max = np.amax(variable)
    var_min = np.amin(variable)

    bin_dict = {}
    quant_dict = {}
    quant_error_dict = {}
    mu_dict = {}
    mu_error_dict = {}

    divisor = 1
    if var_max - var_min <= 100:
        divisor=10
    if var_max - var_min > 100:
        divisor=100
        if var_min-divisor < 0:
            var_min=0
            if var_max > 2000:
                var_min=100
        #Round max to nearest 100
        var_max = float(ceil((var_max-divisor)/100.0))*100.
        if var_max > 4000:
            var_max=4000
    print(f"var min: {var_min}, var max: {var_max}, var num: {int((var_max-var_min)/divisor)+1}, divisor: {divisor}")
    var_bins = np.linspace(var_min,var_max, num=int((var_max-var_min)/divisor)+1)
    print(f"VAR BINS: {var_bins}")
    quantile, mu = [], []

    for i, bin in enumerate(tqdm(var_bins)):
        if i >= len(var_bins)-1:
            continue
        temp_dir = dir[(variable > var_bins[i]) & (variable < var_bins[i+1])]
        temp_pred = pred[(variable > var_bins[i]) & (variable < var_bins[i+1])]
        temp_true = true[(variable > var_bins[i]) & (variable < var_bins[i+1])]
        temp_bin = var_bins[i]+(var_bins[i+1]-var_bins[i])/2
        axes, temp_quantile, temp_quantile_error, temp_mu, temp_mu_error = regression_analysis(from_path=from_path, dirpath=dirpath, combine=combine, true=temp_true, pred=temp_pred, dir=temp_dir, target = target, extra_string = extra_string, save_plots=save_plots)
        for i, axis in enumerate(axes):
            if axis in quant_dict:
                bin_dict[axis].append(temp_bin) 
                quant_dict[axis].append(temp_quantile[i]) 
                quant_error_dict[axis].append(temp_quantile_error[i]) 
                mu_dict[axis].append(temp_mu[i]) 
                mu_error_dict[axis].append(temp_mu_error[i]) 
            else:
                bin_dict[axis] = [temp_bin] 
                quant_dict[axis] = [temp_quantile[i]]
                quant_error_dict[axis] = [temp_quantile_error[i]] 
                mu_dict[axis] = [temp_mu[i]] 
                mu_error_dict[axis] = [temp_mu_error[i]] 
        '''
        if "momentum" in target or "momenta" in target:
            temp_quantile = 100*float(temp_quantile[0])
            temp_mu = 100*float(temp_mu[0])
            quantile.append(temp_quantile)
            mu.append(temp_mu)
        else:
            quantile.append(temp_quantile)
            mu.append(temp_mu)
    if "positions" in target or "directions" in target:
        axes = ['X', 'Y', 'Z', 'Global']
        for i, axis in enumerate(axes):
            if "Global" or "Transverse" or "Longitudinal" in axis:
                print(f"{target} quantile for {axis} axis: {np.array(quantile)[:,i]}")
                print(f"{target} mu for {axis} axis: {np.array(mu)[:,i]}")
    else:
        print(f"{target} Quantile: {quantile}", sep=", ")
        print(f"{target} Mu: {mu}", sep=", ")
        '''
    return bin_dict, quant_dict, quant_error_dict, mu_dict, mu_error_dict




def regression_analysis(from_path=True, dirpath='outputs', combine=True, true=None, pred=None, dir=None, target = 'directions', extra_string = "", save_plots=False, plot_path='outputs/'):
     '''
     saves 2 plots for each of the 3 axes (x,y,z):
     1) scatter plot of predicted vs true position
     2) histogram of redsiduals with guassian fit
     if combine = True this is done for both classes in the same
     plot but if combine = False there is a set of plots for each class
     residuals corner plot is only outputted when combine = True
     uncertainties are likely calculated wrong
     this is meant to be called in a notebook. if you would like to call it
     in the command line you likely wanna change plt.show() to save figure instead
     the default is to read from specified file path but you can also set from_path 
     to False and pass the specific true and predicted position arrays directly
     '''
     plt.style.use('ggplot')

     if from_path: 
         # read in true positions and model predicted positions
         true_positions = np.load(dirpath + "true_positions.npy")
         pred_positions = np.load(dirpath + "pred_positions.npy")


         # put data into natural units
         tp, pp = [], []
         for p, t in zip(pred_positions, true_positions):
             tp.append(un_normalize(t))
             pp.append(un_normalize(p))
         true_positions = np.array(tp)
         pred_positions = np.array(pp)

         # read in true class and model predicted class
         true_class = np.load(dirpath + "true_class.npy")
         pred_class = np.load(dirpath + "pred_class.npy")

     # just defining some basics
     vertex_axis = ['X','Y','Z']
     if "positions" in target:
         vertex_axis.append('Global')
         vertex_axis.append('Longitudinal')
         vertex_axis.append('Transverse')
     if "directions" in target:
         vertex_axis.append('Angle')
         vertex_axis.append('Longitudinal Angle')
         vertex_axis.append('Transverse Angle')
     if "momentum" in target or "energies" in target or "momenta" in target:
         vertex_axis = ['Global']
     xlimit = 200
     ylimit = 200
     if "directions" in target:
        xlimit=15.
        ylimit=15.
     if "momentum" in target or "energies" in target or "momenta" in target:
        xlimit=4
        ylimit=4
     line = np.linspace(-xlimit, xlimit, 10000) 
     residual_lst, residual_lst_wcut, quantile_lst,quantile_error_lst, median_lst, median_error_lst = [], [], [], [], [], []
     color_choices=[
    '#1f77b4', '#ff7f0e', '#2ca02c', 
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5']

     # loop over X, then Y, then Z and show in different colours
     for i in range(len(vertex_axis)): 
         # debug
         print('i-th vertex', vertex_axis[i], 'of len = ', len(vertex_axis))
         #color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
         color = color_choices[i]
         #Add some custom axes
         if 'R' in vertex_axis[i]:
             #true = np.append(true, np.sqrt(np.square(true[:,0] )+ np.square(true[:,1])), axis=1)
             true = np.hstack((true, np.reshape(np.sqrt(np.square(true[:,0] )+ np.square(true[:,1])), (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(np.sqrt(np.square(pred[:,0] )+ np.square(pred[:,1])), (pred.shape[0], 1))))
         #Bit of a hack
         if 'Global' in vertex_axis[i] and "positions" in target:
             #true = np.append(true, np.sqrt(np.square(true[:,0] )+ np.square(true[:,1])), axis=1)
             true = np.hstack((true, np.reshape(np.sqrt(np.square(true[:,0] -pred[:,0])+ np.square(true[:,1] - pred[:,1]) + np.square(true[:,2] - pred[:,2])), (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(np.square(pred[:,0] )*0, (pred.shape[0], 1))))
         if 'Longitudinal' in vertex_axis[i] and "positions" in target:
             total_magnitude_pred, longitudinal_component_pred, _ = math.decompose_along_direction(pred[:,0:3], dir)
             total_magnitude_true, longitudinal_component_true, _ = math.decompose_along_direction(true[:,0:3], dir)
             temp_true = longitudinal_component_true
             temp_pred = longitudinal_component_pred
             true = np.hstack((true, np.reshape(temp_true, (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(temp_pred, (pred.shape[0], 1))))
         if 'Transverse' in vertex_axis[i] and "positions" in target:
             total_magnitude_pred, _, transverse_component_pred = math.decompose_along_direction(pred[:,0:3], dir)
             total_magnitude_true, _, transverse_component_true = math.decompose_along_direction(true[:,0:3], dir)
             temp_true = transverse_component_true
             temp_pred = transverse_component_pred
             true = np.hstack((true, np.reshape(temp_true, (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(temp_pred, (pred.shape[0], 1))))

         if "Angle" in vertex_axis[i]:
             unit_vector_pred = np.transpose(np.array([np.divide(pred[:,0],np.linalg.norm(pred[:,0:3], axis=1)), np.divide(pred[:,1],np.linalg.norm(pred[:,0:3], axis=1)), np.divide(pred[:,2],np.linalg.norm(pred[:,0:3], axis=1))]) )
             unit_vector_true = np.transpose(np.array([np.divide(true[:,0],np.linalg.norm(true[:,0:3], axis=1)), np.divide(true[:,1],np.linalg.norm(true[:,0:3], axis=1)), np.divide(true[:,2],np.linalg.norm(true[:,0:3], axis=1))]) )
             #unit_vector_true = np.divide(true[:,0:3],np.linalg.norm(true[:,0:3], axis=1))
             temp_true = np.einsum('...j,...j',unit_vector_pred, unit_vector_true)
             temp_true = np.clip(temp_true, -1.0, 1.0)
             temp_true = np.arccos(temp_true)
             temp_true = np.degrees(temp_true)
             true = np.hstack((true, np.reshape(temp_true, (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(np.zeros(temp_true.shape), (pred.shape[0], 1))))

         if not combine: 
             labels = "muon", "electron"

             true_target = {}
             pred_target = {}

             true_target['0'] = true[true_class==0]
             true_target['1'] = true[true_class==1]
             pred_target['0'] = pred[true_class==0]
             pred_target['1'] = pred[true_class==1]

             pred_mis_id = {}
             pred_cor_id = {}
             true_mis_id = {}
             true_cor_id = {}

             pred_mis_id['0'] = pred_target['0'][np.around(pred_class[true_class==0],0) != 0] 
             pred_cor_id['0'] = pred_target['0'][np.around(pred_class[true_class==0],0) == 0]
             pred_mis_id['1'] = pred_target['1'][np.around(pred_class[true_class==1],0) != 1]
             pred_cor_id['1'] = pred_target['1'][np.around(pred_class[true_class==1],0) == 1]
             true_mis_id['0'] = true_target['0'][np.around(pred_class[true_class==0],0) != 0] 
             true_cor_id['0'] = true_target['0'][np.around(pred_class[true_class==0],0) == 0]
             true_mis_id['1'] = true_target['1'][np.around(pred_class[true_class==1],0) != 1]
             true_cor_id['1'] = true_target['1'][np.around(pred_class[true_class==1],0) == 1]

             for j in range(len(labels)):

                 plt.figure(figsize=(5,5))
                 plt.scatter(true_target[str(j)][:,i], pred_target[str(j)][:,i], alpha=0.05, s=0.1, color=color, label='correct classification')
                 plt.scatter(true_mis_id[str(j)][:,i], pred_mis_id[str(j)][:,i], alpha=0.2, s=0.1, color='black', label='incorrect classification')
                 plt.plot(line, line, '--', color='black', alpha=0.5)


                 plt.xlim(-xlimit,xlimit) 
                 plt.ylim(-ylimit,ylimit)

                 plt.title(f'Event Vertex for {vertex_axis[i]} Axis - {labels[j]}')
                 plt.xlabel('True Position [cm]')
                 plt.ylabel('Predicted Position [cm]')
                 plt.legend()
                 plt.show()

                 # calculate residuals 
                 residuals = true_pos[str(j)][:,i] - pred_pos[str(j)][:,i] 

                 # create cut that we are interested in this value +/- from 0
                 cut = 1600
                 residuals_cut = []
                 for r in range(len(residuals)):
                     if -cut < residuals[r] <  cut:
                         residuals_cut.append(residuals[r])
                 numerical_std = np.std(residuals_cut) 
                 numerical_mean = np.mean(residuals_cut) 

                 yhist, xhist, _ = plt.hist(residuals_cut, bins=100, alpha=0.7, color=color)
                 '''
                 popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=[40, 0, 70])    
                 perr = np.sqrt(np.diag(pcov))
                 plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

                 # round numbers
                 mu = round(popt[1], 2)
                 mu_uncert = round(perr[1], 2)
                 std = round(popt[2], 2)
                 std_uncert = round(perr[2], 2)
                 '''

                 plt.text(0.08, 0.82, '$\mu$ = {} $\pm$ {} [cm] \n$\sigma$ = {} $\pm$ {} [cm]\n Num $\mu$ = {} \n $\sigma = {}'.format(mu, mu_uncert, std, std_uncert, round(numerical_mean,2), round(numerical_std,2)), fontsize=10, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

                 plt.xlim(-cut, cut)

                 plt.title(f'Event Vertex for {vertex_axis[i]} Axis - {labels[j]} (correct and incorrect predicted class)')
                 plt.xlabel('true - predicted [cm]')
                 plt.ylabel('count')
                 plt.yscale('log')
                 plt.show()

         else:

             plt.figure(figsize=(5,5))
             if len(vertex_axis) == 1:
                plt.scatter(true[:], pred[:], alpha=0.05, s=0.1, color=color)
             else:
                plt.scatter(true[:,i], pred[:,i], alpha=0.05, s=0.1, color=color)
             plt.plot(line, line, '--', color='black', alpha=0.5)



             if "positions"in target and "Global" in vertex_axis[i]:
                plt.xlim(0,500) 
             elif "directions"in target and "Angle" in vertex_axis[i]:
                plt.xlim(0,15) 
             else:
                plt.xlim(-xlimit,xlimit) 
                plt.ylim(-ylimit,ylimit)

             plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
             unit = '[cm]'
             if "directions" in target:
                 unit=''
             if "momentum" in target or "momenta" in target:
                 unit='[MeV]'
             plt.xlabel('True ' + target + ' ' + unit)
             plt.ylabel('Predicted ' + target + ' ' + unit)
             #plt.show()
             plt.clf()
             if len(vertex_axis) == 1:
                residuals = (true - pred)/true
             elif "Global" in vertex_axis[i] and "positions" in target:
                residuals = true[:,i]
             else:
                residuals = true[:,i] - pred[:,i] 
             cut = 500
             if "momentum" in target or "energies" in target or "momenta" in target:
                 cut = 10
             residuals_cut = [] 
             for r in range(len(residuals)):
                 if -cut < residuals[r] <  cut:
                     residuals_cut.append(residuals[r])
            
            #  print('heres your residuals', residuals)
                     
             #numerical_std = np.std(residuals_cut) 
             #Is actually median
             numerical_median = np.median(residuals_cut) 
             #Compute mean for error
             mean_error = np.std(residuals_cut)/np.sqrt(len(residuals_cut))
             mean_to_median = np.sqrt(3.14*((2.*float(len(residuals_cut))+1.)/(4.*(len(residuals_cut)))))
             median_error = mean_error * mean_to_median
             if ("positions"in target and "Global" in vertex_axis[i]) or ("directions" in target and "Angle" in vertex_axis[i]):
                 quantile = np.quantile(residuals_cut,0.68)
                 quantile_error = mquantiles_cimj(residuals_cut, prob=0.68)
                 #quantile_error = (quantile-np.ravel(quantile_error)[0], np.ravel(quantile_error)[1]-quantile)
             else:
                (numerical_bot_quantile, numerical_top_quantile) = np.quantile(residuals_cut, [0.159,0.841])
                #numerical_top_quantile = np.quantile(residuals_cut, 0.841)
                #quantile = (np.abs((numerical_median-numerical_bot_quantile))+np.abs((numerical_median-numerical_top_quantile)))/2
                quantile = np.quantile(np.abs(residuals_cut),0.68)
                quantile_error = mquantiles_cimj(np.abs(residuals_cut), prob=[0.68])
                #quantile_error = (((quantile_error[1][0] - quantile_error[0][0])/2),((quantile_error[1][1] - quantile_error[0][1])/2))
             quantile_error = float((quantile_error[1]-quantile_error[0])/2.)

             yhist, xhist, _ = plt.hist(residuals_cut, bins=100, alpha=0.7, color=color, range=[-xlimit, xlimit])
             p0 = [40, 0, 70] 
             if "directions" in target or "momentum" in target or "energies" in target or "momenta" in target:
                 p0 = [40, 0, 0.03]
             '''
             popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=p0)    
             perr = np.sqrt(np.diag(pcov))
             mu = round(popt[1], dec_to_round)
             mu_uncert = round(perr[1], dec_to_round)
             std = round(popt[2], dec_to_round)
             std_uncert = round(perr[2], dec_to_round)
             '''

             #plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

             dec_to_round = 2
             if "directions" in target or "energies" in target or "momentum" in target or "momenta":
                 dec_to_round = 5
             # round numbers

             if "momenta" in target or "momentum" in target:
                plt.text(0.6, 0.7, '{} \n $\mu$ = {:.5f} $\pm${} {} \n Quant. = {:.5f} $\pm${} {}'.format(extra_string, round(numerical_median,dec_to_round), round_sig(median_error), unit, round(quantile, dec_to_round), round_sig(quantile_error), unit), fontsize=9, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
             else:
                plt.text(0.6, 0.7, '{} \n $\mu$ = {:.2f} $\pm${} {} \n Quant. = {:.2f} $\pm${} {}'.format(extra_string, round(numerical_median,dec_to_round), round_sig(median_error), unit, round(quantile, dec_to_round), round_sig(quantile_error), unit), fontsize=9, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

             if "positions"in target and "Global" in vertex_axis[i]:
                plt.xlim(0,xlimit) 
             elif "directions"in target and "Angle" in vertex_axis[i]:
                plt.xlim(0,15) 
             else:
                plt.xlim(-xlimit, xlimit)

             plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
             if "energies" in target or "momentum" in target or "momenta" in target:
                plt.xlabel('(true - predicted)/true ')
             else:
                plt.xlabel('true - predicted ' + unit)
             plt.ylabel('count')
             plt.yscale('log')
             plt.ylim((0.1, 200000))
             if save_plots:
                plt.savefig(f"{plot_path}/pred_{target}_{vertex_axis[i]}_{extra_string}.png")
             
             # debug
             plt.savefig(f"/data/thoriba/t2k/plots/plots_regression_test4/pred_{target}_{vertex_axis[i]}_{extra_string}.png")

             residual_lst.append(residuals)
             residual_lst_wcut.append(residuals_cut)

             quantile_lst.append(quantile)
             quantile_error_lst.append(quantile_error)
             median_lst.append(numerical_median)
             median_error_lst.append(median_error)
             plt.clf()

     if False and combine:
         residuals = np.array(residual_lst)
         labels = ['X', 'Y', 'Z']
         if "momentum" in target:
             labels = ['Global']
         figure = corner(residuals.T, bins=50,  labels=labels, range=[(-cut,cut), (-cut,cut), (-cut,cut)]) 
         plt.show()

     plt.clf()
     
     print('some debugging')
     print('true', np.array(true).shape)
     print('residuals', np.array(residuals).shape)
     print('residuals cut', np.array(residuals_cut).shape)
     


     return vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst


# def compute_residuals(from_path=True, dirpath='outputs', combine=True, true=None, pred=None, dir=None, target = 'directions', extra_string = "", save_plots=False, plot_path='outputs/'):


def compute_residuals(from_path=True, dirpath='outputs', combine=True, true=None, pred=None, dir=None, target = 'directions', extra_string = "", save_plots=False, plot_path='outputs/', v_axis='Longitudinal'):
     '''
     return the residuals for specific axis specified in `vertex_axis`
     '''
     plt.style.use('ggplot')

    #  print('hello')


     if from_path: 
        print('from_path not supported')
        return
         # read in true positions and model predicted positions
        #  true_positions = np.load(dirpath + "true_positions.npy")
        #  pred_positions = np.load(dirpath + "pred_positions.npy")


        #  # put data into natural units
        #  tp, pp = [], []
        #  for p, t in zip(pred_positions, true_positions):
        #      tp.append(un_normalize(t))
        #      pp.append(un_normalize(p))
        #  true_positions = np.array(tp)
        #  pred_positions = np.array(pp)

        #  # read in true class and model predicted class
        #  true_class = np.load(dirpath + "true_class.npy")
        #  pred_class = np.load(dirpath + "pred_class.npy")

     # just defining some basics
     vertex_axis = ['X','Y','Z']
     if "positions" in target:
         vertex_axis.append('Global')
         vertex_axis.append('Longitudinal')
         vertex_axis.append('Transverse')
     if "directions" in target:
         vertex_axis.append('Angle')
         vertex_axis.append('Longitudinal Angle')
         vertex_axis.append('Transverse Angle')
     if "momentum" in target or "energies" in target or "momenta" in target:
         vertex_axis = ['Global']
     xlimit = 200
     ylimit = 200
     if "directions" in target:
        xlimit=15.
        ylimit=15.
     if "momentum" in target or "energies" in target or "momenta" in target:
        xlimit=4
        ylimit=4
     line = np.linspace(-xlimit, xlimit, 10000) 
     residual_lst, residual_lst_wcut, quantile_lst,quantile_error_lst, median_lst, median_error_lst = [], [], [], [], [], []
     color_choices=[
    '#1f77b4', '#ff7f0e', '#2ca02c', 
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5']

     if v_axis not in vertex_axis:
         print(f'Your specified axis {v_axis} is not available')
         return
     
     # loop over X, then Y, then Z and show in different colours
     for i in range(len(vertex_axis)): 
         # debug
        #  print('i-th vertex', vertex_axis[i], 'of len = ', len(vertex_axis))
         #color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
         color = color_choices[i]
         #Add some custom axes
         if 'R' in vertex_axis[i]:
             #true = np.append(true, np.sqrt(np.square(true[:,0] )+ np.square(true[:,1])), axis=1)
             true = np.hstack((true, np.reshape(np.sqrt(np.square(true[:,0] )+ np.square(true[:,1])), (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(np.sqrt(np.square(pred[:,0] )+ np.square(pred[:,1])), (pred.shape[0], 1))))
         #Bit of a hack
         if 'Global' in vertex_axis[i] and "positions" in target:
             #true = np.append(true, np.sqrt(np.square(true[:,0] )+ np.square(true[:,1])), axis=1)
             true = np.hstack((true, np.reshape(np.sqrt(np.square(true[:,0] -pred[:,0])+ np.square(true[:,1] - pred[:,1]) + np.square(true[:,2] - pred[:,2])), (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(np.square(pred[:,0] )*0, (pred.shape[0], 1))))
         if 'Longitudinal' in vertex_axis[i] and "positions" in target:
             total_magnitude_pred, longitudinal_component_pred, _ = math.decompose_along_direction(pred[:,0:3], dir)
             total_magnitude_true, longitudinal_component_true, _ = math.decompose_along_direction(true[:,0:3], dir)
             temp_true = longitudinal_component_true
             temp_pred = longitudinal_component_pred
             true = np.hstack((true, np.reshape(temp_true, (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(temp_pred, (pred.shape[0], 1))))
         if 'Transverse' in vertex_axis[i] and "positions" in target:
             total_magnitude_pred, _, transverse_component_pred = math.decompose_along_direction(pred[:,0:3], dir)
             total_magnitude_true, _, transverse_component_true = math.decompose_along_direction(true[:,0:3], dir)
             temp_true = transverse_component_true
             temp_pred = transverse_component_pred
             true = np.hstack((true, np.reshape(temp_true, (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(temp_pred, (pred.shape[0], 1))))

         if "Angle" in vertex_axis[i]:
             unit_vector_pred = np.transpose(np.array([np.divide(pred[:,0],np.linalg.norm(pred[:,0:3], axis=1)), np.divide(pred[:,1],np.linalg.norm(pred[:,0:3], axis=1)), np.divide(pred[:,2],np.linalg.norm(pred[:,0:3], axis=1))]) )
             unit_vector_true = np.transpose(np.array([np.divide(true[:,0],np.linalg.norm(true[:,0:3], axis=1)), np.divide(true[:,1],np.linalg.norm(true[:,0:3], axis=1)), np.divide(true[:,2],np.linalg.norm(true[:,0:3], axis=1))]) )
             #unit_vector_true = np.divide(true[:,0:3],np.linalg.norm(true[:,0:3], axis=1))
             temp_true = np.einsum('...j,...j',unit_vector_pred, unit_vector_true)
             temp_true = np.clip(temp_true, -1.0, 1.0)
             temp_true = np.arccos(temp_true)
             temp_true = np.degrees(temp_true)
             true = np.hstack((true, np.reshape(temp_true, (true.shape[0], 1))))
             pred = np.hstack((pred, np.reshape(np.zeros(temp_true.shape), (pred.shape[0], 1))))

         if not combine: 
             labels = "muon", "electron"

             true_target = {}
             pred_target = {}

             true_target['0'] = true[true_class==0]
             true_target['1'] = true[true_class==1]
             pred_target['0'] = pred[true_class==0]
             pred_target['1'] = pred[true_class==1]

             pred_mis_id = {}
             pred_cor_id = {}
             true_mis_id = {}
             true_cor_id = {}

             pred_mis_id['0'] = pred_target['0'][np.around(pred_class[true_class==0],0) != 0] 
             pred_cor_id['0'] = pred_target['0'][np.around(pred_class[true_class==0],0) == 0]
             pred_mis_id['1'] = pred_target['1'][np.around(pred_class[true_class==1],0) != 1]
             pred_cor_id['1'] = pred_target['1'][np.around(pred_class[true_class==1],0) == 1]
             true_mis_id['0'] = true_target['0'][np.around(pred_class[true_class==0],0) != 0] 
             true_cor_id['0'] = true_target['0'][np.around(pred_class[true_class==0],0) == 0]
             true_mis_id['1'] = true_target['1'][np.around(pred_class[true_class==1],0) != 1]
             true_cor_id['1'] = true_target['1'][np.around(pred_class[true_class==1],0) == 1]

             for j in range(len(labels)):

                 plt.figure(figsize=(5,5))
                 plt.scatter(true_target[str(j)][:,i], pred_target[str(j)][:,i], alpha=0.05, s=0.1, color=color, label='correct classification')
                 plt.scatter(true_mis_id[str(j)][:,i], pred_mis_id[str(j)][:,i], alpha=0.2, s=0.1, color='black', label='incorrect classification')
                 plt.plot(line, line, '--', color='black', alpha=0.5)


                 plt.xlim(-xlimit,xlimit) 
                 plt.ylim(-ylimit,ylimit)

                 plt.title(f'Event Vertex for {vertex_axis[i]} Axis - {labels[j]}')
                 plt.xlabel('True Position [cm]')
                 plt.ylabel('Predicted Position [cm]')
                 plt.legend()
                 plt.show()

                 # calculate residuals 
                 residuals = true_pos[str(j)][:,i] - pred_pos[str(j)][:,i] 

                 # create cut that we are interested in this value +/- from 0
                 cut = 1600
                 residuals_cut = []
                 for r in range(len(residuals)):
                     if -cut < residuals[r] <  cut:
                         residuals_cut.append(residuals[r])
                 numerical_std = np.std(residuals_cut) 
                 numerical_mean = np.mean(residuals_cut) 

                 yhist, xhist, _ = plt.hist(residuals_cut, bins=100, alpha=0.7, color=color)
                 '''
                 popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=[40, 0, 70])    
                 perr = np.sqrt(np.diag(pcov))
                 plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

                 # round numbers
                 mu = round(popt[1], 2)
                 mu_uncert = round(perr[1], 2)
                 std = round(popt[2], 2)
                 std_uncert = round(perr[2], 2)
                 '''

                 plt.text(0.08, 0.82, '$\mu$ = {} $\pm$ {} [cm] \n$\sigma$ = {} $\pm$ {} [cm]\n Num $\mu$ = {} \n $\sigma = {}'.format(mu, mu_uncert, std, std_uncert, round(numerical_mean,2), round(numerical_std,2)), fontsize=10, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

                 plt.xlim(-cut, cut)

                 plt.title(f'Event Vertex for {vertex_axis[i]} Axis - {labels[j]} (correct and incorrect predicted class)')
                 plt.xlabel('true - predicted [cm]')
                 plt.ylabel('count')
                 plt.yscale('log')
                 plt.show()

         else:

            #  plt.figure(figsize=(5,5))
            #  if len(vertex_axis) == 1:
            #     plt.scatter(true[:], pred[:], alpha=0.05, s=0.1, color=color)
            #  else:
            #     plt.scatter(true[:,i], pred[:,i], alpha=0.05, s=0.1, color=color)
            #  plt.plot(line, line, '--', color='black', alpha=0.5)
            #  if "positions"in target and "Global" in vertex_axis[i]:
            #     plt.xlim(0,500) 
            #  elif "directions"in target and "Angle" in vertex_axis[i]:
            #     plt.xlim(0,15) 
            #  else:
            #     plt.xlim(-xlimit,xlimit) 
            #     plt.ylim(-ylimit,ylimit)

            #  plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
            #  unit = '[cm]'
            #  if "directions" in target:
            #      unit=''
            #  if "momentum" in target or "momenta" in target:
            #      unit='[MeV]'
            #  plt.xlabel('True ' + target + ' ' + unit)
            #  plt.ylabel('Predicted ' + target + ' ' + unit)
            #  #plt.show()
            #  plt.clf()
             if len(vertex_axis) == 1:
                residuals = (true - pred)/true
             elif "Global" in vertex_axis[i] and "positions" in target:
                residuals = true[:,i]
            #  elif vertex_axis[i] == 'Longitudinal' and 'positions' in target:
            #      print('returing long residuals')
            #      return true[:,i] - pred[:,i] 
             else:
                residuals = true[:,i] - pred[:,i] 

             if v_axis == vertex_axis[i]:
                 print(f'Computed resisduals along {v_axis}. Returning the residual vector')
                 return residuals
             

             cut = 500
             if "momentum" in target or "energies" in target or "momenta" in target:
                 cut = 10
             residuals_cut = [] 
             for r in range(len(residuals)):
                 if -cut < residuals[r] <  cut:
                     residuals_cut.append(residuals[r])
            
            #  print('heres your residuals', residuals)
                     
             #numerical_std = np.std(residuals_cut) 
             #Is actually median
             numerical_median = np.median(residuals_cut) 
             #Compute mean for error
             mean_error = np.std(residuals_cut)/np.sqrt(len(residuals_cut))
             mean_to_median = np.sqrt(3.14*((2.*float(len(residuals_cut))+1.)/(4.*(len(residuals_cut)))))
             median_error = mean_error * mean_to_median
             if ("positions"in target and "Global" in vertex_axis[i]) or ("directions" in target and "Angle" in vertex_axis[i]):
                 quantile = np.quantile(residuals_cut,0.68)
                 quantile_error = mquantiles_cimj(residuals_cut, prob=0.68)
                 #quantile_error = (quantile-np.ravel(quantile_error)[0], np.ravel(quantile_error)[1]-quantile)
             else:
                (numerical_bot_quantile, numerical_top_quantile) = np.quantile(residuals_cut, [0.159,0.841])
                #numerical_top_quantile = np.quantile(residuals_cut, 0.841)
                #quantile = (np.abs((numerical_median-numerical_bot_quantile))+np.abs((numerical_median-numerical_top_quantile)))/2
                quantile = np.quantile(np.abs(residuals_cut),0.68)
                quantile_error = mquantiles_cimj(np.abs(residuals_cut), prob=[0.68])
                #quantile_error = (((quantile_error[1][0] - quantile_error[0][0])/2),((quantile_error[1][1] - quantile_error[0][1])/2))
             quantile_error = float((quantile_error[1]-quantile_error[0])/2.)

             yhist, xhist, _ = plt.hist(residuals_cut, bins=100, alpha=0.7, color=color, range=[-xlimit, xlimit])
             p0 = [40, 0, 70] 
             if "directions" in target or "momentum" in target or "energies" in target or "momenta" in target:
                 p0 = [40, 0, 0.03]
             '''
             popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=p0)    
             perr = np.sqrt(np.diag(pcov))
             mu = round(popt[1], dec_to_round)
             mu_uncert = round(perr[1], dec_to_round)
             std = round(popt[2], dec_to_round)
             std_uncert = round(perr[2], dec_to_round)
             '''

             #plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

             dec_to_round = 2
             if "directions" in target or "energies" in target or "momentum" in target or "momenta":
                 dec_to_round = 5
             # round numbers

            #  if "momenta" in target or "momentum" in target:
            #     plt.text(0.6, 0.7, '{} \n $\mu$ = {:.5f} $\pm${} {} \n Quant. = {:.5f} $\pm${} {}'.format(extra_string, round(numerical_median,dec_to_round), round_sig(median_error), unit, round(quantile, dec_to_round), round_sig(quantile_error), unit), fontsize=9, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
            #  else:
            #     plt.text(0.6, 0.7, '{} \n $\mu$ = {:.2f} $\pm${} {} \n Quant. = {:.2f} $\pm${} {}'.format(extra_string, round(numerical_median,dec_to_round), round_sig(median_error), unit, round(quantile, dec_to_round), round_sig(quantile_error), unit), fontsize=9, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

            #  if "positions"in target and "Global" in vertex_axis[i]:
            #     plt.xlim(0,xlimit) 
            #  elif "directions"in target and "Angle" in vertex_axis[i]:
            #     plt.xlim(0,15) 
            #  else:
            #     plt.xlim(-xlimit, xlimit)

            #  plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
            #  if "energies" in target or "momentum" in target or "momenta" in target:
            #     plt.xlabel('(true - predicted)/true ')
            #  else:
            #     plt.xlabel('true - predicted ' + unit)
            #  plt.ylabel('count')
            #  plt.yscale('log')
            #  plt.ylim((0.1, 200000))
            #  if save_plots:
            #     plt.savefig(f"{plot_path}/pred_{target}_{vertex_axis[i]}_{extra_string}.png")
             
             # debug
            #  plt.savefig(f"/data/thoriba/t2k/plots/plots_regression_test4/pred_{target}_{vertex_axis[i]}_{extra_string}.png")

            #  residual_lst.append(residuals)
            #  residual_lst_wcut.append(residuals_cut)

            #  quantile_lst.append(quantile)
            #  quantile_error_lst.append(quantile_error)
            #  median_lst.append(numerical_median)
            #  median_error_lst.append(median_error)
            #  plt.clf()

            #  if vertex_axis[i] == 'Longitudinal':
            #      print('long residual shape', np.array(residuals).shape)
            #      print('long residual head', np.array(residuals)[:20])
            #      return residuals
     print(f'Could not compute {vertex_axis}. Returning None')
     return None
    #  if False and combine:
    #      residuals = np.array(residual_lst)
    #      labels = ['X', 'Y', 'Z']
    #      if "momentum" in target:
    #          labels = ['Global']
    #      figure = corner(residuals.T, bins=50,  labels=labels, range=[(-cut,cut), (-cut,cut), (-cut,cut)]) 
    #      plt.show()

    #  plt.clf()
     
    #  print('some debugging')
    #  print('true', np.array(true).shape)
    #  print('residuals', np.array(residuals).shape)
    #  print('residuals cut', np.array(residuals_cut).shape)
     


    #  return vertex_axis, quantile_lst, quantile_error_lst, median_lst, median_error_lst




def un_normalize(data, x_bounds=(-1600,1600), y_bounds=(-1600,1600), z_bounds=(-1600,1600)):
     '''
     putting normalized data for x,y,z values from regression model back into natural units
     '''
     bounds = [x_bounds, y_bounds, z_bounds]
     for i in range(3):
         data[i] = (data[i])*bounds[i][1]

     return data

def gaussian(x, a, mean, sigma):
     return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))



