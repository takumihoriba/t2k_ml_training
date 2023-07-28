import numpy as np

import h5py

from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile
from analysis.utils.plotting import plot_legend
from analysis.utils.binning import get_binning
import analysis.utils.math as math

def efficiency_plots(inputPath, arch_name, newest_directory, plot_output, label=None):

    # retrieve test indices
    idx = np.array(sorted(np.load(str(newest_directory)+'/indices.npy')))

    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(inputPath, "r")
    #print(list(hy.keys()))
    angles = np.array(hy['angles'])[idx].squeeze() 
    labels = np.array(hy['labels'])[idx].squeeze() 
    veto = np.array(hy['veto'])[idx].squeeze()
    energies = np.array(hy['energies'])[idx].squeeze()
    positions = np.array(hy['positions'])[idx].squeeze()
    directions = math.direction_from_angles(angles)

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
    #print('basic cuts = ', basic_cuts)
    # set class labels and decrease values within labels to match either 0 or 1 
    e_label = [0]
    mu_label = [1]
    #labels = [x - 1 for x in labels]

    # get the bin indices and edges for parameters
    polar_binning = get_binning(np.cos(angles[:,0]), 10, -1, 1)
    az_binning = get_binning(angles[:,1]*180/np.pi, 10, -180, 180)
    mom_binning = get_binning(momentum, 10)
    dwall_binning = get_binning(dwall, 10)
    towall_binning = get_binning(towall, 10)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    stride1 = newest_directory
    run_result = [WatChMaLClassification(stride1, 'test', labels, idx, basic_cuts, color="blue", linestyle='-')]
    
    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.99876
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_fixed_efficiency(e_label, mu_label, muon_efficiency, select_labels = mu_label, selection = basic_cuts)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, e_polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_az_fig, az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_mom_fig, mom_ax = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_az_fig, az_ax = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_mom_fig, mom_ax = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_towall_fig, towall_ax = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)

    # save plots of effiency as a function of specific parameters
    e_polar_fig.savefig(plot_output + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(plot_output + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(plot_output + 'e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(plot_output + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(plot_output + 'e_towall_efficiency.png', format='png')

    mu_polar_fig.savefig(plot_output + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(plot_output + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(plot_output + 'mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(plot_output + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(plot_output + 'mu_towall_efficiency.png', format='png')

    # remove comment for ROC curves of single run 
    return run_result[0]
