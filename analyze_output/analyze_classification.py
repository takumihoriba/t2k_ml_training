import numpy as np

from analyze_output.utils.plotting import softmax_plots
from analyze_output.utils.math import get_cherenkov_threshold

import os

import h5py

from analysis.classification import WatChMaLClassification
from analysis.classification import plot_efficiency_profile, plot_rocs
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

def analyze_classification(settings):


    # retrieve test indices.
    # must use correct indices for dataset.
    idx = np.array(sorted(np.load(str(settings.mlPath) + "/indices.npy")))
    idx = np.unique(idx)
    softmax = np.array(np.load(str(settings.mlPath) + "/softmax.npy"))
    
    labels_test = np.array(np.load(str(settings.mlPath) + "/labels.npy"))
    #positions_array = np.array(np.load(str(newest_directory) + "/pred_positions.npy"))
    #true_positions_array = np.array(np.load(str(newest_directory) + "/true_positions.npy"))

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


    #softmax_sig = [softmax[label] for label in settings.signalLabels]
    #softmax_bkg = [softmax[label] for label in settings.bkgLabels]

    softmax_e = softmax[labels==1]
    softmax_mu = softmax[labels==0]
    softmax_pi = softmax[labels==2]

    if settings.signalLabels==0:
        softmax_plots([softmax_e[:,1], softmax_e[:,0]+softmax_e[:,2]], ['e-score', 'mu+pi-score'], extra_label='Electrons only', file_path=settings.outputPlotPath)
        softmax_plots([softmax_mu[:,1], softmax_mu[:,0]+softmax_mu[:,2]], ['e-score', 'mu+pi-score'], extra_label='Muons only', file_path=settings.outputPlotPath)
        softmax_plots([softmax_pi[:,1], softmax_pi[:,0]+softmax_pi[:,2]], ['e-score', 'mu+pi-score'], extra_label='Pi+ only', file_path=settings.outputPlotPath)

        softmax_plots([softmax_pi[:,0], softmax_pi[:,2]], ['mu-score', 'pi-score'], extra_label='Pi+ only', file_path=settings.outputPlotPath)
        softmax_plots([softmax_mu[:,0], softmax_mu[:,2]], ['mu-score', 'pi-score'], extra_label='Muon only', file_path=settings.outputPlotPath)

    # softmax_plots([np.log(np.divide(softmax_pi[:,0], softmax_pi[:,2])), np.log(np.divide(softmax_mu[:,0], softmax_mu[:,2]))], ['Pi+ only','Muons only'], extra_label='ln mu over pi', range=[-100,100])


    #Save ids and rootfiles to compare to fitqun, after applying cuts
    ml_hash = get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)


    softmax_sum = np.sum(softmax,axis=1)
    print(f"SOFTMAX SUM: {np.amin(softmax_sum)}")

    # calculate additional parameters 
    towall = math.towall(positions, angles, tank_axis = 2)
    dwall = math.dwall(positions, tank_axis = 2)
    momentum = math.momentum_from_energy(energies, labels)
    ml_cheThr = list(map(get_cherenkov_threshold, labels))
    ml_visible_energy = energies - ml_cheThr



    do_fitqun=False
    if os.path.isfile(str(settings.fitqunPath) + "/fitqun_combine.hy") and settings.doFiTQun:
        print("Running fiTQun")
        do_fitqun=True
        fitqun_discr, fitqun_pi_discr, fitqun_labels, fitqun_mom, fitqun_hash = read_fitqun_file(str(settings.fitqunPath) + "/fitqun_combine.hy", plotting=False, regression=False)
        print(f'len idx: {len(idx)}, len fitqun: {len(fitqun_discr)}')
        fitqun_idx = np.array(range(len(fitqun_discr)))
        fitqun_hash = np.array(fitqun_hash)[fitqun_idx].squeeze()
        fitqun_discr = fitqun_discr[fitqun_idx].squeeze() 
        if 2 in settings.bkgLabels:
            fitqun_pi_discr = fitqun_pi_discr[fitqun_idx].squeeze() 
        fitqun_labels = fitqun_labels[fitqun_idx].squeeze() 
        fitqun_mom = fitqun_mom[fitqun_idx].squeeze() 
        fitqun_energy = math.energy_from_momentum(fitqun_mom, fitqun_labels)
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_energy - fitqun_cheThr

        #Get the ids that are in both ML and fitqun samples
        intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, assume_unique=True, return_indices=True)
        print(f'intersect: {intersect.shape}, comm1: {comm1.shape}, comm2: {comm2.shape}')
        print("len comm1:", len(comm1))
        print('len comm2:', len(comm2))

        fitqun_matched_energies = energies[comm2]
        fitqun_dwall = dwall[comm2]
        fitqun_az = (angles[:,1]*180/np.pi)[comm2]
        fitqun_polar = np.cos(angles[:,0])[comm2] 
        fitqun_towall = towall[comm2]
        fitqun_discr = fitqun_discr[comm1]
        fitqun_pi_discr = fitqun_pi_discr[comm1]
        fitqun_labels = fitqun_labels[comm1]
        fitqun_idx = fitqun_idx[comm2]
        fitqun_mom = momentum[comm2]
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_matched_energies - fitqun_cheThr

        temp = np.abs(fitqun_labels[fitqun_towall > 100]-fitqun_discr[fitqun_towall > 100])

        print(f"fitqun e- avg towall > 100): {1-np.sum(temp[fitqun_labels[fitqun_towall > 100]==1])/len(temp[fitqun_labels[fitqun_towall > 100]==1])}")
        print(f"fitqun mu- avg (towall > 100): {1-np.sum(temp[fitqun_labels[fitqun_towall > 100]==0])/len(temp[fitqun_labels[fitqun_towall > 100]==0])}")

        print("fitqun towall matrix", fitqun_towall > 100)
        print("fitqun towall shape", fitqun_towall.shape)
        print("fitqun label vector", fitqun_labels)
        print("fitqun label vector shape", fitqun_labels.shape)
        print("fitqun_labels",fitqun_labels[fitqun_towall > 100]==1)
        print("fitqun_labels shape",(fitqun_labels[fitqun_towall > 100]==1).shape)
        print("fitqun_labels len",temp[fitqun_labels[fitqun_towall > 100]==1])
        
    # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
    nhit_cut = nhits > 0 #25
    towall_cut = towall > 100
    # veto_cut = (veto == 0)
    hy_electrons = (labels == 0)
    hy_muons = (labels == 2)
    print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
    basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)

    # set class labels and decrease values within labels to match either 0 or 1 
    if type(settings.signalLabels) == list:
        e_label = [settings.signalLabels]
    else:
        e_label = [settings.signalLabels]
    if type(settings.bkgLabels) == list:
        mu_label = settings.bkgLabels
    else:
        mu_label = settings.bkgLabels
    #labels = [x - 1 for x in labels]

    # get the bin indices and edges for parameters
    polar_binning = get_binning(np.cos(angles[:,0]), 10, -1, 1)
    az_binning = get_binning(angles[:,1]*180/np.pi, 10, -180, 180)
    mom_binning = get_binning(momentum, 9, minimum=100, maximum=1000)
    visible_energy_binning = get_binning(ml_visible_energy, 10, minimum=0, maximum=1000)
    dwall_binning = get_binning(dwall, 15, minimum=0, maximum=1600)
    towall_binning = get_binning(towall, 30, minimum=0, maximum=3000)

    # create watchmal classification object to be used as runs for plotting the efficiency relative to event angle  
    stride1 = settings.mlPath
    run_result = [WatChMaLClassification(stride1, 'test', labels, idx, basic_cuts, color="blue", linestyle='-')]
    # print(f"UNIQUE IN LABLES: {np.unique(fitqun_labels, return_counts=True)}")

    
    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    #Fitqun
    if do_fitqun:
        fitqun_hy_electrons = (fitqun_labels == 0)
        fitqun_hy_muons = (fitqun_labels == 2)
        fitqun_basic_cuts = ((fitqun_hy_electrons | fitqun_hy_muons) & (fitqun_towall > 100))
        fitqun_mom_binning = get_binning(fitqun_mom, 9, minimum=100, maximum=1000)
        fitqun_ve_binning = get_binning(fitqun_visible_energy, 10, minimum=0, maximum=1000)
        fitqun_towall_binning = get_binning(fitqun_towall, 30, minimum=0, maximum=3000)
        fitqun_az_binning = get_binning(fitqun_az, 10, minimum=-180, maximum=180)
        fitqun_polar_binning = get_binning(fitqun_polar, 10, minimum=-1, maximum=1)
        fitqun_run_result = [WatChMaLClassification(stride1, 'test', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        (fitqun_run_result[0]).selection = fitqun_basic_cuts
        fitqun_run_result[0].cut = fitqun_discr.astype(bool)

        fitqun_pi_run_result = [WatChMaLClassification(stride1, 'ML', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        (fitqun_pi_run_result[0]).selection = fitqun_basic_cuts
        fitqun_pi_run_result[0].cut = fitqun_pi_discr.astype(bool)

    if True:
        #For electron/muon
        #ALSO have to change e/mu definitions for fitqun and ML
        #cut_pi_discr = fitqun_discr[fitqun_basic_cuts]
        #fitqun_pi_eff = np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 1])/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==1])
        #fitqun_bkg_rej = np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 0]-1))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==0])
        #For mu/pi+
        # fitqun_pi_eff = .5
        # fitqun_bkg_rej = .5
        # print(do_fitqun)
        if do_fitqun:
            cut_pi_discr = fitqun_pi_discr[fitqun_basic_cuts]
            fitqun_pi_eff = np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] ==0])/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==0])
            fitqun_bkg_rej = np.abs(np.sum(cut_pi_discr[fitqun_labels[fitqun_basic_cuts] == 2]-1))/len(cut_pi_discr[fitqun_labels[fitqun_basic_cuts]==2])
        print(f"fiTQun signal efficiency: {fitqun_pi_eff}, fiTQun bkg rejection: {fitqun_bkg_rej}")
        fig_roc, ax_roc = plot_rocs(run_result, e_label, mu_label, selection=basic_cuts, x_label="Electron Tagging Efficiency", y_label="Muon Rejection",
                legend='best', mode='rejection', fitqun=(fitqun_pi_eff, fitqun_bkg_rej), label='ML')
        fig_roc.savefig(settings.outputPlotPath + '/ml_pi_roc.png', format='png')

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.961
    muon_efficiency = 1 - muon_rejection
    print(e_label)
    for r in run_result:
        r.cut_with_constant_binned_efficiency(e_label, mu_label, 0.98, binning = visible_energy_binning, select_labels = e_label)

    label='eMuPi Training'


    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    e_polar_fig, polar_ax_e = plot_efficiency_profile(run_result, polar_binning, select_labels=e_label, x_label="Cosine of Zenith", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_az_fig, az_ax_e = plot_efficiency_profile(run_result, az_binning, select_labels=e_label, x_label="Azimuth [Degree]", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_mom_fig, mom_ax_e = plot_efficiency_profile(run_result, mom_binning, select_labels=e_label, x_label="True Momentum", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_ve_fig, ve_ax_e = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=e_label, x_label="True Visible Energy [MeV]", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=e_label, x_label="Distance from Detector Wall [cm]", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    e_towall_fig, towall_ax_e = plot_efficiency_profile(run_result, towall_binning, select_labels=e_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Muon Signal PID Efficiency [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        e_mom_fig_fitqun, mom_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_mom_binning, select_labels=e_label, x_label="fiTQun e Momentum [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_ve_fig_fitqun, ve_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_ve_binning, select_labels=e_label, x_label="fiTQun Visible energy [MeV]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_towall_fig_fitqun, towall_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_towall_binning, select_labels=e_label, x_label="Truth toWall [cm]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)
        e_az_fig_fitqun, az_ax_fitqun_e = plot_efficiency_profile(fitqun_pi_run_result, fitqun_az_binning, select_labels=e_label, x_label="Truth Azimuth [deg]", y_label="fiTQun Electron Signal PID Efficiency [%]", errors=True, x_errors=False, label='fitqun'+label)

    # plot signal efficiency against true momentum, dwall, towall, zenith, azimuth
    mu_polar_fig, polar_ax_mu = plot_efficiency_profile(run_result, polar_binning, select_labels=mu_label, x_label="Cosine of Zenith", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_az_fig, az_ax_mu = plot_efficiency_profile(run_result, az_binning, select_labels=mu_label, x_label="Azimuth [Degree]", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_mom_fig, mom_ax_mu = plot_efficiency_profile(run_result, mom_binning, select_labels=mu_label, x_label="True Momentum", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_ve_fig, ve_ax_mu = plot_efficiency_profile(run_result, visible_energy_binning, select_labels=mu_label, x_label="True Visible Energy [MeV]", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_towall_fig, towall_ax_mu = plot_efficiency_profile(run_result, towall_binning, select_labels=mu_label, x_label="Distance to Wall Along Particle Direction [cm]  ", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    mu_dwall_fig, dwall_ax = plot_efficiency_profile(run_result, dwall_binning, select_labels=mu_label, x_label="Distance from Detector Wall [cm]", y_label="Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
    if do_fitqun:
        mu_mom_fig_fitqun, mom_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_mom_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Pi+ Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_ve_fig_fitqun, ve_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_ve_binning, select_labels=mu_label, x_label="fiTQun e Momentum", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_towall_fig_fitqun, towall_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_towall_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)
        mu_az_fig_fitqun, az_ax_fitqun_mu = plot_efficiency_profile(fitqun_pi_run_result, fitqun_az_binning, select_labels=mu_label, x_label="Towall [cm]", y_label="fiTQun Muon Background Miss-PID [%]", errors=True, x_errors=False, label=label)

    # save plots of effiency as a function of specific parameters
    e_polar_fig.savefig(settings.outputPlotPath + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(settings.outputPlotPath + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(settings.outputPlotPath + 'e_momentum_efficiency.png', format='png')
    e_ve_fig.savefig(settings.outputPlotPath + 'e_ve_efficiency.png', format='png')
    if do_fitqun:
        e_mom_fig_fitqun.savefig(settings.outputPlotPath + 'fitqun_e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(settings.outputPlotPath + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(settings.outputPlotPath + 'e_towall_efficiency.png', format='png')

    mu_polar_fig.savefig(settings.outputPlotPath + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(settings.outputPlotPath + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(settings.outputPlotPath + 'mu_momentum_efficiency.png', format='png')
    mu_ve_fig.savefig(settings.outputPlotPath + 'mu_ve_efficiency.png', format='png')
    if do_fitqun:
        mu_mom_fig_fitqun.savefig(settings.outputPlotPath + 'fitqun_mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(settings.outputPlotPath + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(settings.outputPlotPath + 'mu_towall_efficiency.png', format='png')

    if do_fitqun:
        plot_fitqun_comparison(settings.outputPlotPath, mom_ax_e, mom_ax_fitqun_e, mom_ax_mu, mom_ax_fitqun_mu, 'mom_combine', 'Truth Momentum [MeV]')
        plot_fitqun_comparison(settings.outputPlotPath, ve_ax_e, ve_ax_fitqun_e, ve_ax_mu, ve_ax_fitqun_mu, 've_combine', 'Truth Visible Energy [MeV]')
        plot_fitqun_comparison(settings.outputPlotPath, towall_ax_e, towall_ax_fitqun_e, towall_ax_mu, towall_ax_fitqun_mu, 'towall_combine', 'Towall [cm]')
        #plot_fitqun_comparison(plot_output, az_ax_e, az_ax_fitqun_e, az_ax_mu, az_ax_fitqun_mu, 'az_combine', 'Truth Azimuth [deg]')



    # remove comment for ROC curves of single run 
    return run_result[0]