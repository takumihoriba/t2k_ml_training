import numpy as np

import os

import h5py

from WatChMaL.analysis.classification import WatChMaLClassification
from WatChMaL.analysis.classification import plot_efficiency_profile
from WatChMaL.analysis.utils.plotting import plot_legend
from WatChMaL.analysis.utils.binning import get_binning
from WatChMaL.analysis.utils.fitqun import read_fitqun_file, make_fitqunlike_discr, get_rootfile_eventid_hash, plot_fitqun_comparison
import analysis.utils.math as math

import matplotlib
from matplotlib import pyplot as plt

def get_cherenkov_threshold(label):
    threshold_dict = {0: 160., 1:0.8, 2:211.715}
    return threshold_dict[label]

def efficiency_plots(inputPath, arch_name, newest_directory, plot_output, label=None):
    print('1')

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

    print('2')

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

        # by definition this should work? gone now which is good
        # error persists which makes no sense
        intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, assume_unique=True, return_indices=True)
        print(f'intersect: {intersect}, comm1: {comm1}, comm2: {comm2}')
        print(len(comm1))
        print(len(comm2))

        fitqun_matched_energies = energies[comm2]
        fitqun_dwall = dwall[comm2]
        fitqun_az = (angles[:,1]*180/np.pi)[comm2]
        fitqun_polar = np.cos(angles[:,0])[comm2] 
        fitqun_towall = towall[comm2]
        fitqun_discr = fitqun_discr[comm2]
        fitqun_labels = fitqun_labels[comm2]
        fitqun_idx = fitqun_idx[comm2]
        #fitqun_discr = fitqun_discr[comm1]
        #fitqun_labels = fitqun_labels[comm1]
        #fitqun_idx = fitqun_idx[comm1]
        fitqun_mom = momentum[comm2] # should all fitqun be 1?
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
    muon_rejection = 0.961 # change this from: 0.961 I suppose?
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_constant_binned_efficiency(e_label, mu_label, 0.98557, binning = visible_energy_binning, select_labels = e_label) # line where ml efficiency flat in vis energy is required
        # instead used gloabl efficiency vs fixed effiiency, thows out bad ones? what is the cut needed on confidence or energy, cut on ml output, can force electrons to be really good, but bad muon rejection

    print(f"BAD EVENTS CHECK: {len(run_result[0]._softmaxes[(run_result[0]._softmaxes[:,1] > 0.5) & (labels == 0) & (ml_visible_energy > 900)])}")

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    #plt.clf()
    e_polar_fig, polar_ax_e = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    e_az_fig, az_ax_e = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    e_mom_fig, mom_ax_e = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    e_ve_fig, ve_ax_e = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=e_label, x_label="True Visible Energy [MeV]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        plt.clf()
        e_mom_fig_fitqun, mom_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_mom_binning, select_labels=e_label, x_label="fiTQun e Momentum [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_ve_fig_fitqun, ve_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_ve_binning, select_labels=e_label, x_label="fiTQun Visible energy [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_towall_fig_fitqun, towall_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_towall_binning, select_labels=e_label, x_label="Truth toWall [cm]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_az_fig_fitqun, az_ax_fitqun_e = plot_efficiency_profile(fitqun_run_result, fitqun_az_binning, select_labels=e_label, x_label="Truth Azimuth [deg]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
    plt.clf()
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    e_towall_fig, towall_ax_e = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax_mu = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    mu_az_fig, az_ax_mu = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    mu_mom_fig, mom_ax_mu = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
    mu_ve_fig, ve_ax_mu = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=mu_label, x_label="True Momentum", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    plt.clf()
    if do_fitqun: # why are some of these not saved?
        mu_mom_fig_fitqun, mom_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_mom_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_ve_fig_fitqun, ve_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_ve_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_towall_fig_fitqun, towall_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_towall_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_az_fig_fitqun, az_ax_fitqun_mu = plot_efficiency_profile(fitqun_run_result, fitqun_az_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    plt.clf()
    mu_towall_fig, towall_ax_mu = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    #plt.clf()
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

    plt.clf() # FIGURE OUT WHERE BEST TO PUT THESE
    if do_fitqun:
        plot_fitqun_comparison(plot_output, mom_ax_e, mom_ax_fitqun_e, mom_ax_mu, mom_ax_fitqun_mu, 'mom_combine', 'Truth Momentum [MeV]')
        plot_fitqun_comparison(plot_output, ve_ax_e, ve_ax_fitqun_e, ve_ax_mu, ve_ax_fitqun_mu, 've_combine', 'Truth Visible Energy [MeV]')
        plot_fitqun_comparison(plot_output, towall_ax_e, towall_ax_fitqun_e, towall_ax_mu, towall_ax_fitqun_mu, 'towall_combine', 'Towall [cm]')
        plot_fitqun_comparison(plot_output, az_ax_e, az_ax_fitqun_e, az_ax_mu, az_ax_fitqun_mu, 'az_combine', 'Truth Azimuth [deg]')



    # remove comment for ROC curves of single run 
    return run_result[0]


