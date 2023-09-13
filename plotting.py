import numpy as np

import os

import h5py

from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile
from analysis.utils.plotting import plot_legend
from analysis.utils.binning import get_binning
from analysis.utils.fitqun import read_fitqun_file, make_fitqunlike_discr
import analysis.utils.math as math

import matplotlib
from matplotlib import pyplot as plt

def efficiency_plots(inputPath, arch_name, newest_directory, plot_output, label=None):

    # retrieve test indices
    idx = np.array(sorted(np.load(str(newest_directory) + "/indices.npy")))
    softmax = np.array(np.load(str(newest_directory) + "/softmax.npy"))
    do_fitqun=False
    if os.path.isfile(str(newest_directory) + "/fitqun_combine.hy"):
        do_fitqun-True
        fitqun_discr, fitqun_labels, fitqun_mom = read_fitqun_file(str(newest_directory) + "/fitqun_combine.hy")
        print(f'len idx: {len(idx)}, len fitqun: {len(fitqun_discr)}')
        fitqun_idx = np.array(range(len(fitqun_discr)))
        fitqun_discr = fitqun_discr[fitqun_idx].squeeze() 
        fitqun_labels = fitqun_labels[fitqun_idx].squeeze() 
        fitqun_mom = fitqun_mom[fitqun_idx].squeeze() 
    

    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(inputPath, "r")
    angles = np.array(hy['angles'])[idx].squeeze() 
    labels = np.array(hy['labels'])[idx].squeeze() 
    print(labels)
    veto = np.array(hy['veto'])[idx].squeeze()
    energies = np.array(hy['energies'])[idx].squeeze()
    positions = np.array(hy['positions'])[idx].squeeze()
    directions = math.direction_from_angles(angles)

    #make_fitqunlike_discr(softmax, energies, labels)

    # calculate number of hits 
    events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
    nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()

    # calculate additional parameters 
    towall = math.towall(positions, angles, tank_axis = 2)
    dwall = math.dwall(positions, tank_axis = 2)
    momentum = math.momentum_from_energy(energies, labels)

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
    dwall_binning = get_binning(dwall, 16, minimum=0, maximum=1600)
    towall_binning = get_binning(towall, 10)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    stride1 = newest_directory
    run_result = [WatChMaLClassification(stride1, 'test', labels, idx, basic_cuts, color="blue", linestyle='-')]
    
    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    #Fitqun
    if do_fitqun:
        fitqun_hy_electrons = (fitqun_labels == 1)
        fitqun_hy_muons = (fitqun_labels == 0)
        fitqun_basic_cuts = (fitqun_hy_electrons | fitqun_hy_muons)
        fitqun_mom_binning = get_binning(fitqun_mom, 10, minimum=0, maximum=1000)
        fitqun_run_result = [WatChMaLClassification(stride1, 'test', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        fitqun_run_result[0].cut = fitqun_discr

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.99
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_constant_binned_efficiency(e_label, mu_label, muon_efficiency, binning = mom_binning, select_labels = mu_label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, e_polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_az_fig, az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_mom_fig, mom_ax_e = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        e_mom_fig_fitqun, mom_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_mom_binning, select_labels=e_label, x_label="fiTQun e Momentum [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_az_fig, az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_mom_fig, mom_ax_mu = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        mu_mom_fig_fitqun, mom_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_mom_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)

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
        plt.clf()
        plt.figure(e_mom_fig.number)
        print(mom_ax_e.lines[0].get_xdata())
        print(mom_ax_e.lines[0].get_ydata())
        print(mom_ax_fitqun_e.lines[0].get_ydata())
        print(mom_ax_mu.lines[0].get_ydata())
        print(mom_ax_fitqun_mu.lines[0].get_ydata())
        plt.figure(e_mom_fig_fitqun.number)
        plt.savefig(plot_output + 'e_mom_combine.png', format='png')

    # remove comment for ROC curves of single run 
    return run_result[0]
