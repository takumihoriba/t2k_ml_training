import numpy as np

import os

import h5py

from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile
from analysis.utils.plotting import plot_legend
from analysis.utils.binning import get_binning
from analysis.utils.fitqun import read_fitqun_file, make_fitqunlike_discr, get_rootfile_eventid_hash, plot_fitqun_comparison
import analysis.utils.math as math

import matplotlib
from matplotlib import pyplot as plt

import torch
import torch.onnx
from WatChMaL.watchmal.model.resnet import resnet101

from corner import corner
from scipy.optimize import curve_fit

dummy_path = '/fast_scratch_2/aferreira/t2k/ml/data/oct20_combine_flatE_oneClass/dec20_label0_justRegression_resnet101/20092023-101855/'

def get_cherenkov_threshold(label):
    threshold_dict = {0: 160., 1:0.8, 2:211.715}
    return threshold_dict[label]

def efficiency_plots(inputPath, arch_name, newest_directory, plot_output, label=None):

    # retrieve test indices
    idx = np.array(sorted(np.load(str(newest_directory) + "/indices.npy")))
    softmax = np.array(np.load(str(newest_directory) + "/softmax.npy"))

    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(inputPath, "r")
    angles = np.array(hy['angles'])[idx].squeeze() 
    labels = np.array(hy['labels'])[idx].squeeze() 
    veto = np.array(hy['veto'])[idx].squeeze()
    energies = np.array(hy['energies'])[idx].squeeze()
    positions = np.array(hy['positions'])[idx].squeeze()
    directions = math.direction_from_angles(angles)
    rootfiles = np.array(hy['root_files'])[idx].squeeze()
    event_ids = np.array(hy['event_ids'])[idx].squeeze()
    ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)

    # calculate number of hits 
    events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
    nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()

    # calculate additional parameters 
    towall = math.towall(positions, angles, tank_axis = 2)
    dwall = math.dwall(positions, tank_axis = 2)
    momentum = math.momentum_from_energy(energies, labels)
    ml_cheThr = list(map(get_cherenkov_threshold, labels))
    ml_visible_energy = energies - ml_cheThr



    do_fitqun=False
    if os.path.isfile(str(newest_directory) + "/fitqun_combine.hy"):
        print("Running fiTQun")
        do_fitqun=True
        fitqun_discr, fitqun_labels, fitqun_mom, fitqun_hash = read_fitqun_file(str(newest_directory) + "/fitqun_combine.hy")
        print(f'len idx: {len(idx)}, len fitqun: {len(fitqun_discr)}')
        fitqun_idx = np.array(range(len(fitqun_discr)))
        fitqun_hash = np.array(fitqun_hash)[fitqun_idx].squeeze()
        fitqun_discr = fitqun_discr[fitqun_idx].squeeze() 
        fitqun_labels = fitqun_labels[fitqun_idx].squeeze() 
        fitqun_mom = fitqun_mom[fitqun_idx].squeeze() 
        fitqun_energy = math.energy_from_momentum(fitqun_mom, fitqun_labels)
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_energy - fitqun_cheThr

        intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, assume_unique=True, return_indices=True)
        print(f'intersect: {intersect}, comm1: {comm1}, comm2: {comm2}')
        print(len(comm1))
        print(len(comm2))

        fitqun_matched_energies = energies[comm2]
        fitqun_dwall = dwall[comm2]
        fitqun_az = (angles[:,1]*180/np.pi)[comm2]
        fitqun_polar = np.cos(angles[:,0])[comm2] 
        fitqun_towall = towall[comm2]
        fitqun_discr = fitqun_discr[comm1]
        fitqun_labels = fitqun_labels[comm1]
        fitqun_idx = fitqun_idx[comm1]
        fitqun_mom = momentum[comm2]
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_matched_energies - fitqun_cheThr
        





    #make_fitqunlike_discr(softmax, energies, labels)


    # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
    nhit_cut = nhits > 0 #25
    # veto_cut = (veto == 0)
    hy_electrons = (labels == 1)
    hy_muons = (labels == 0)
    basic_cuts = ((hy_electrons | hy_muons) & nhit_cut)

    # set class labels and decrease values within labels to match either 0 or 1 
    e_label = [1]
    mu_label = [0]
    #labels = [x - 1 for x in labels]

    # get the bin indices and edges for parameters
    polar_binning = get_binning(np.cos(angles[:,0]), 10, -1, 1)
    az_binning = get_binning(angles[:,1]*180/np.pi, 10, -180, 180)
    mom_binning = get_binning(momentum, 9, minimum=100, maximum=1000)
    visible_energy_binning = get_binning(ml_visible_energy, 10, minimum=0, maximum=1000)
    dwall_binning = get_binning(dwall, 15, minimum=0, maximum=1600)
    towall_binning = get_binning(towall, 30, minimum=0, maximum=3000)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    stride1 = newest_directory
    run_result = [WatChMaLClassification(stride1, 'test', labels, idx, basic_cuts, color="blue", linestyle='-')]

    
    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    #Fitqun
    if do_fitqun:
        print("2")
        fitqun_hy_electrons = (fitqun_labels == 1)
        fitqun_hy_muons = (fitqun_labels == 0)
        fitqun_basic_cuts = (fitqun_hy_electrons | fitqun_hy_muons)
        fitqun_mom_binning = get_binning(fitqun_mom, 9, minimum=100, maximum=1000)
        fitqun_ve_binning = get_binning(fitqun_visible_energy, 10, minimum=0, maximum=1000)
        fitqun_towall_binning = get_binning(fitqun_towall, 30, minimum=0, maximum=3000)
        fitqun_az_binning = get_binning(fitqun_az, 10, minimum=-180, maximum=180)
        fitqun_polar_binning = get_binning(fitqun_polar, 10, minimum=-1, maximum=1)
        fitqun_run_result = [WatChMaLClassification(stride1, 'test', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        print(f'fitqun_discr: {fitqun_discr}')
        fitqun_run_result[0].cut = fitqun_discr.astype(np.bool)

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.961
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_constant_binned_efficiency(e_label, mu_label, 0.98557, binning = visible_energy_binning, select_labels = e_label)

    print(f"BAD EVENTS CHECK: {len(run_result[0]._softmaxes[(run_result[0]._softmaxes[:,1] > 0.5) & (labels == 0) & (ml_visible_energy > 900)])}")

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, polar_ax_e = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_az_fig, az_ax_e = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_mom_fig, mom_ax_e = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_ve_fig, ve_ax_e = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=e_label, x_label="True Visible Energy [MeV]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        e_mom_fig_fitqun, mom_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_mom_binning, select_labels=e_label, x_label="fiTQun e Momentum [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_ve_fig_fitqun, ve_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_ve_binning, select_labels=e_label, x_label="fiTQun Visible energy [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_towall_fig_fitqun, towall_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_towall_binning, select_labels=e_label, x_label="Truth toWall [cm]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_az_fig_fitqun, az_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_az_binning, select_labels=e_label, x_label="Truth Azimuth [deg]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_towall_fig, towall_ax_e = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax_mu = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_az_fig, az_ax_mu = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_mom_fig, mom_ax_mu = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_ve_fig, ve_ax_mu = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        mu_mom_fig_fitqun, mom_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_mom_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_ve_fig_fitqun, ve_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_ve_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_towall_fig_fitqun, towall_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_towall_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_az_fig_fitqun, az_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_az_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_towall_fig, towall_ax_mu = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)

    # save plots of effiency as a function of specific parameters
    e_polar_fig.savefig(plot_output + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(plot_output + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(plot_output + 'e_momentum_efficiency.png', format='png')
    if do_fitqun:
        e_mom_fig_fitqun.savefig(plot_output + 'fitqun_e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(plot_output + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(plot_output + 'e_towall_efficiency.png', format='png')

    mu_polar_fig.savefig(plot_output + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(plot_output + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(plot_output + 'mu_momentum_efficiency.png', format='png')
    if do_fitqun:
        mu_mom_fig_fitqun.savefig(plot_output + 'fitqun_mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(plot_output + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(plot_output + 'mu_towall_efficiency.png', format='png')

    if do_fitqun:
        plot_fitqun_comparison(plot_output, mom_ax_e, mom_ax_fitqun_e, mom_ax_mu, mom_ax_fitqun_mu, 'mom_combine', 'Truth Momentum [MeV]')
        plot_fitqun_comparison(plot_output, ve_ax_e, ve_ax_fitqun_e, ve_ax_mu, ve_ax_fitqun_mu, 've_combine', 'Truth Visible Energy [MeV]')
        plot_fitqun_comparison(plot_output, towall_ax_e, towall_ax_fitqun_e, towall_ax_mu, towall_ax_fitqun_mu, 'towall_combine', 'Towall [cm]')
        plot_fitqun_comparison(plot_output, az_ax_e, az_ax_fitqun_e, az_ax_mu, az_ax_fitqun_mu, 'az_combine', 'Truth Azimuth [deg]')



    # remove comment for ROC curves of single run 
    return run_result[0]


def network_diagram():
    '''
    code to produce onnx file of model network architecture that can then
    be uploaded to https://netron.app/ in order to get network diagram
    
    the default input size is likey wrong since plot doesn't look quite right

    currently just generates this for resnet 101 but should work with all models

    can be run locally, does not require significant compute

    I have put resulting resnet_onnx.png in t2k_ml_training/data/
    '''
    torch_model = resnet101(num_input_channels=2, num_output_channels=2)
    # input size outputted from first forward pass of resnet
    # (but with batch size of 256 instead of 1)
    torch_input = torch.randn(1, 2, 198, 150) 
    torch.onnx.export(torch_model, torch_input, 'resnet.onnx', opset_version=11)

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

def regression_analysis(dirpath=dummy_path, combine=True):
    '''
    saves 2 plots for each of the 3 axes (x,y,z):
    1) scatter plot of predicted vs true position
    2) histogram of redsiduals with guassian fit

    if combine = True this is done for both classes in the same
    plot but if combine = False there is a set of plots for each class

    residuals corner plot is only outputted when combine = True

    this is meant to be called in a notebook. if you would like to call it
    in the command line you likely wanna change plt.show() to save figure instead
    '''
    plt.style.use('ggplot')

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
    vertex_axis = ['X', 'Y', 'Z']
    line = np.linspace(-1600, 1600, 10) 
    residual_lst, residual_lst_wcut = [], []

    # loop over X, then Y, then Z and show in different colours
    for i in range(len(vertex_axis)): 
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]

        if not combine: 
            labels = "muon", "electron"

            true_pos = {}
            pred_pos = {}

            true_pos['0'] = true_positions[true_class==0]
            true_pos['1'] = true_positions[true_class==1]
            pred_pos['0'] = pred_positions[true_class==0]
            pred_pos['1'] = pred_positions[true_class==1]

            pred_mis_id = {}
            pred_cor_id = {}
            true_mis_id = {}
            true_cor_id = {}

            pred_mis_id['0'] = pred_pos['0'][np.around(pred_class[true_class==0],0) != 0] 
            pred_cor_id['0'] = pred_pos['0'][np.around(pred_class[true_class==0],0) == 0]
            pred_mis_id['1'] = pred_pos['1'][np.around(pred_class[true_class==1],0) != 1]
            pred_cor_id['1'] = pred_pos['1'][np.around(pred_class[true_class==1],0) == 1]
            true_mis_id['0'] = true_pos['0'][np.around(pred_class[true_class==0],0) != 0] 
            true_cor_id['0'] = true_pos['0'][np.around(pred_class[true_class==0],0) == 0]
            true_mis_id['1'] = true_pos['1'][np.around(pred_class[true_class==1],0) != 1]
            true_cor_id['1'] = true_pos['1'][np.around(pred_class[true_class==1],0) == 1]

            for j in range(len(labels)):

                plt.figure(figsize=(5,5))
                plt.scatter(true_pos[str(j)][:,i], pred_pos[str(j)][:,i], alpha=0.05, s=0.1, color=color, label='correct classification')
                plt.scatter(true_mis_id[str(j)][:,i], pred_mis_id[str(j)][:,i], alpha=0.5, s=0.1, color='black', label='incorrect classification')
                plt.plot(line, line, '--', color='black', alpha=0.5)

                plt.xlim(-2000,2000) 
                plt.ylim(-2000,2000)
                
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

                yhist, xhist, _ = plt.hist(residuals_cut, bins=50, alpha=0.7, color=color)
                popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=[40, 0, 70])    
                perr = np.sqrt(np.diag(pcov))
                plt.plot(x, gaussian(x, *popt), alpha=1, color='black', label='guassian fit')

                # round numbers
                mu = round(popt[1], 2)
                mu_uncert = round(perr[1], 2)
                std = round(popt[2], 2)
                std_uncert = round(perr[2], 2)

                plt.text(0.08, 0.82, '$\mu$ = {} $\pm$ {} [cm] \n\n$\sigma$ = {} $\pm$ {} [cm]'.format(mu, mu_uncert, std, std_uncert), fontsize=10, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

                plt.xlim(-cut, cut)

                plt.title(f'Event Vertex for {vertex_axis[i]} Axis - {labels[j]} (correct and incorrect predicted class)')
                plt.xlabel('true - predicted [cm]')
                plt.ylabel('count')
                plt.show()

        else:

            plt.figure(figsize=(5,5))
            plt.scatter(true_positions[:,i], pred_positions[:,i], alpha=0.05, s=0.1, color=color)
            plt.plot(line, line, '--', color='black', alpha=0.5)

            plt.xlim(-2000,2000) 
            plt.ylim(-2000,2000)
            
            plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
            plt.xlabel('True Position [cm]')
            plt.ylabel('Predicted Position [cm]')
            plt.legend()
            plt.show()

            residuals = true_positions[:,i] - pred_positions[:,i] 
            cut = 1600
            residuals_cut = [] 
            for r in range(len(residuals)):
                if -cut < residuals[r] <  cut:
                    residuals_cut.append(residuals[r])

            yhist, xhist, _ = plt.hist(residuals_cut, bins=50, alpha=0.7, color=color)
            popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=[40, 0, 70])    
            perr = np.sqrt(np.diag(pcov))

            plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

            # round numbers
            mu = round(popt[1], 2)
            mu_uncert = round(perr[1], 2)
            std = round(popt[2], 2)*-1
            std_uncert = round(perr[2], 2)

            plt.text(0.08, 0.82, '$\mu$ = {} $\pm$ {} [cm] \n\n$\sigma$ = {} $\pm$ {} [cm]'.format(mu, mu_uncert, std, std_uncert), fontsize=10, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

            plt.xlim(-cut, cut)

            plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
            plt.xlabel('true - predicted [cm]')
            plt.ylabel('count')
            plt.show()

            residual_lst.append(residuals)
            residual_lst_wcut.append(residuals_cut)

            residuals = np.array(residual_lst)
            figure = corner.corner(residuals.T, bins=50,  labels=['X', 'Y', 'Z'], range=[(-cut,cut), (-cut,cut), (-cut,cut)]) 
            plt.show()