import numpy as np

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


from corner import corner
from scipy.optimize import curve_fit

import WatChMaL.analysis.utils.fitqun as fq


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

def get_cherenkov_threshold(label):
    threshold_dict = {0: 160., 1:0.8, 2:211.715}
    return threshold_dict[label]

def efficiency_plots(inputPath, arch_name, newest_directory, plot_output, label=None):

    print("HERE")
    fitqun_regression_results()
    exit
    do_regression=True
    do_pi=False

    # retrieve test indices
    idx = np.array(sorted(np.load(str(newest_directory) + "/indices.npy")))
    idx = np.unique(idx)
    softmax = np.array(np.load(str(newest_directory) + "/softmax.npy"))
    #positions_array = np.array(np.load(str(newest_directory) + "/pred_positions.npy"))
    #true_positions_array = np.array(np.load(str(newest_directory) + "/true_positions.npy"))

    # grab relevent parameters from hy file and only keep the values corresponding to those in the test set
    hy = h5py.File(inputPath, "r")
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


    if do_regression:
        #Apply cuts
        print("WARNING: ONLY LOOKING AT ELECTRON EVENTS")
        event_ids = event_ids[(nhits> 100) & (labels==1)]
        print(f'len roofiles before cut: {rootfiles.shape}')
        rootfiles = rootfiles[(nhits> 100) & (labels==1)]
        print(f'len roofiles after cut: {rootfiles.shape}')
        positions = positions[(nhits> 100) & (labels==1)]
        angles = angles[(nhits> 100) & (labels==1)]

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
    if os.path.isfile(str(newest_directory) + "/fitqun_combine.hy"):
        print("Running fiTQun")
        do_fitqun=True
        if do_regression:
            (fitqun_discr, fitqun_labels, fitqun_mom, fitqun_hash) , ((fitqun_mu_1rpos, fitqun_e_1rpos, fitqun_mu_1rdir, fitqun_e_1rdir, fitqun_mu_1rmom, fitqun_e_1rmom)) = read_fitqun_file(str(newest_directory) + "/fitqun_combine.hy", plotting=True, regression=do_regression)
        else:
            fitqun_discr, fitqun_pi_discr, fitqun_labels, fitqun_mom, fitqun_hash = read_fitqun_file(str(newest_directory) + "/fitqun_combine.hy", plotting=True, regression=do_regression)
        print(f'len idx: {len(idx)}, len fitqun: {len(fitqun_discr)}')
        fitqun_idx = np.array(range(len(fitqun_discr)))
        fitqun_hash = np.array(fitqun_hash)[fitqun_idx].squeeze()
        fitqun_discr = fitqun_discr[fitqun_idx].squeeze() 
        if do_pi:
            fitqun_pi_discr = fitqun_pi_discr[fitqun_idx].squeeze() 
        fitqun_labels = fitqun_labels[fitqun_idx].squeeze() 
        fitqun_mom = fitqun_mom[fitqun_idx].squeeze() 
        if do_regression:
            #fitqun_e_1rpos = np.reshape(fitqun_e_1rpos,(int(fitqun_e_1rpos.shape[0]/3), 3))
            fitqun_e_1rpos = fitqun_e_1rpos[fitqun_idx].squeeze() 
            #fitqun_mu_1rpos = np.reshape(fitqun_mu_1rpos,(int(fitqun_mu_1rpos.shape[0]/3), 3))
            fitqun_mu_1rpos = fitqun_mu_1rpos[fitqun_idx].squeeze() 
        fitqun_energy = math.energy_from_momentum(fitqun_mom, fitqun_labels)
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_energy - fitqun_cheThr

        #Get the ids that are in both ML and fitqun samples
        intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, assume_unique=True, return_indices=True)
        print(f'intersect: {intersect.shape}, comm1: {comm1.shape}, comm2: {comm2.shape}')
        print(len(comm1))
        print(len(comm2))

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
        if do_regression:
            fitqun_e_1rpos = fitqun_e_1rpos[comm1]
            fitqun_mu_1rpos = fitqun_mu_1rpos[comm1]
        fitqun_cheThr = list(map(get_cherenkov_threshold, fitqun_labels))
        fitqun_visible_energy = fitqun_matched_energies - fitqun_cheThr

        temp = np.abs(fitqun_labels[fitqun_towall > 100]-fitqun_discr[fitqun_towall > 100])

        print(f"fitqun e- avg towall > 100): {1-np.sum(temp[fitqun_labels[fitqun_towall > 100]==1])/len(temp[fitqun_labels[fitqun_towall > 100]==1])}")
        print(f"fitqun mu- avg (towall > 100): {1-np.sum(temp[fitqun_labels[fitqun_towall > 100]==0])/len(temp[fitqun_labels[fitqun_towall > 100]==0])}")


        

    if do_regression:
        print(f"e 1r pos: {fitqun_e_1rpos[:,0]}")
        print(f"weird fitquns x: {len(fitqun_e_1rpos[:,0][np.abs(fitqun_e_1rpos[:,0]) > 2000])}")
        print(f"positions true: {positions[comm2][:,0]}")
        print(f"fitqun idx: {comm1}")
        print(f"event idx: {comm2}")

        if do_fitqun:
            diff = (fitqun_e_1rpos[:,0][np.abs(fitqun_e_1rpos[:,0]) < 2000] - positions[comm2][:,0][np.abs(fitqun_e_1rpos[:,0]) < 2000])
            print(f"diff: {np.amax(fitqun_e_1rpos[:,0])}")
            print(f'Mean diff: {np.mean(diff)}, std: {np.std(diff)}')





    #make_fitqunlike_discr(softmax, energies, labels)


    # apply cuts, as of right now it should remove any events with zero pmt hits (no veto cut)
    nhit_cut = nhits > 0 #25
    towall_cut = towall > 100
    # veto_cut = (veto == 0)
    hy_electrons = (labels == 0)
    hy_muons = (labels == 2)
    print(f"hy_electrons: {hy_electrons.shape}, hy_muons: {hy_muons.shape}, nhit_cut: {nhit_cut.shape}, towall_cut: {towall_cut.shape}")
    basic_cuts = ((hy_electrons | hy_muons) & nhit_cut & towall_cut)

    # set class labels and decrease values within labels to match either 0 or 1 
    e_label = [0]
    mu_label = [2]
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
    print(f"UNIQUE IN LABLES: {np.unique(fitqun_labels, return_counts=True)}")

    
    # for single runs and then can plot the ROC curves with it 
    #run = [WatChMaLClassification(newest_directory, 'title', labels, idx, basic_cuts, color="blue", linestyle='-')]

    #Fitqun
    if do_fitqun:
        print("2")
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
        fitqun_run_result[0].cut = fitqun_discr.astype(np.bool)

        fitqun_pi_run_result = [WatChMaLClassification(stride1, 'test', fitqun_labels, fitqun_idx, fitqun_basic_cuts, color="blue", linestyle='-')]
        (fitqun_pi_run_result[0]).selection = fitqun_basic_cuts
        fitqun_pi_run_result[0].cut = fitqun_pi_discr.astype(np.bool)

    if False:
        fig_roc, ax_roc = plot_rocs(run_result, e_label, mu_label, selection=basic_cuts, x_label="Electron Tagging Efficiency", y_label="Muon Rejection",
                legend='best', mode='rejection', add_fitqun=False)
        fig_roc.savefig(plot_output + 'ml_roc.png', format='png')

    # calculate the thresholds that reject 99.9% of muons and apply cut to all events
    muon_rejection = 0.961
    muon_efficiency = 1 - muon_rejection
    for r in run_result:
        r.cut_with_constant_binned_efficiency(e_label, mu_label, 0.98, binning = visible_energy_binning, select_labels = e_label)


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
    e_polar_fig.savefig(plot_output + 'e_polar_efficiency.png', format='png')
    e_az_fig.savefig(plot_output + 'e_azimuthal_efficiency.png', format='png')
    e_mom_fig.savefig(plot_output + 'e_momentum_efficiency.png', format='png')
    e_ve_fig.savefig(plot_output + 'e_ve_efficiency.png', format='png')
    if do_fitqun:
        e_mom_fig_fitqun.savefig(plot_output + 'fitqun_e_momentum_efficiency.png', format='png')
    e_dwall_fig.savefig(plot_output + 'e_dwall_efficiency.png', format='png')
    e_towall_fig.savefig(plot_output + 'e_towall_efficiency.png', format='png')

    mu_polar_fig.savefig(plot_output + 'mu_polar_efficiency.png', format='png')
    mu_az_fig.savefig(plot_output + 'mu_azimuthal_efficiency.png', format='png')
    mu_mom_fig.savefig(plot_output + 'mu_momentum_efficiency.png', format='png')
    mu_ve_fig.savefig(plot_output + 'mu_ve_efficiency.png', format='png')
    if do_fitqun:
        mu_mom_fig_fitqun.savefig(plot_output + 'fitqun_mu_momentum_efficiency.png', format='png')
    mu_dwall_fig.savefig(plot_output + 'mu_dwall_efficiency.png', format='png')
    mu_towall_fig.savefig(plot_output + 'mu_towall_efficiency.png', format='png')

    if do_fitqun:
        plot_fitqun_comparison(plot_output, mom_ax_e, mom_ax_fitqun_e, mom_ax_mu, mom_ax_fitqun_mu, 'mom_combine', 'Truth Momentum [MeV]')
        plot_fitqun_comparison(plot_output, ve_ax_e, ve_ax_fitqun_e, ve_ax_mu, ve_ax_fitqun_mu, 've_combine', 'Truth Visible Energy [MeV]')
        plot_fitqun_comparison(plot_output, towall_ax_e, towall_ax_fitqun_e, towall_ax_mu, towall_ax_fitqun_mu, 'towall_combine', 'Towall [cm]')
        #plot_fitqun_comparison(plot_output, az_ax_e, az_ax_fitqun_e, az_ax_mu, az_ax_fitqun_mu, 'az_combine', 'Truth Azimuth [deg]')



    # remove comment for ROC curves of single run 
    return run_result[0]


def regression_analysis(from_path=True, dirpath='outputs', combine=True, true=None, pred=None, target = 'positions', extra_string = ""):
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
     vertex_axis = ['X', 'Y', 'Z']
     if "momentum" in target or "energies" in target:
         vertex_axis = ['Global']
     xlimit = 500
     ylimit = 500
     if "directions" in target:
        xlimit=1.1
        ylimit=1.1
     if "momentum" in target or "energies" in target:
        xlimit=10
        ylimit=10
     line = np.linspace(-xlimit, xlimit, 10000) 
     residual_lst, residual_lst_wcut = [], []

     # loop over X, then Y, then Z and show in different colours
     for i in range(len(vertex_axis)): 
         color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]

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
                 popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=[40, 0, 70])    
                 perr = np.sqrt(np.diag(pcov))
                 plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

                 # round numbers
                 mu = round(popt[1], 2)
                 mu_uncert = round(perr[1], 2)
                 std = round(popt[2], 2)
                 std_uncert = round(perr[2], 2)

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



             plt.xlim(-xlimit,xlimit) 
             plt.ylim(-ylimit,ylimit)

             plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
             unit = '[cm]'
             if "directions" in target:
                 unit=''
             if "momentum" in target:
                 unit='[MeV]'
             plt.xlabel('True ' + target + ' ' + unit)
             plt.ylabel('Predicted ' + target + ' ' + unit)
             #plt.show()
             plt.clf()
             if len(vertex_axis) == 1:
                print(np.amin(true))
                residuals = (true - pred)/true
             else:
                residuals = true[:,i] - pred[:,i] 
             cut = 500
             if "momentum" in target or "energies" in target:
                 cut = 10
             residuals_cut = [] 
             for r in range(len(residuals)):
                 if -cut < residuals[r] <  cut:
                     residuals_cut.append(residuals[r])
             #numerical_std = np.std(residuals_cut) 
             (numerical_bot_quantile, numerical_top_quantile) = np.quantile(residuals_cut, [0.159,0.841])
             #numerical_top_quantile = np.quantile(residuals_cut, 0.841)
             numerical_mean = np.mean(residuals_cut) 
             quantile = (np.abs((numerical_mean-numerical_bot_quantile))+np.abs((numerical_mean-numerical_top_quantile)))/2

             yhist, xhist, _ = plt.hist(residuals_cut, bins=100, alpha=0.7, color=color)
             p0 = [40, 0, 70] 
             if "directions" in target or "momentum" in target or "energies" in target:
                 p0 = [40, 0, 0.03]
             popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=p0)    
             perr = np.sqrt(np.diag(pcov))

             plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

             dec_to_round = 2
             if "directions" in target or "energies" in target or "momentum" in target:
                 dec_to_round = 5
             # round numbers
             mu = round(popt[1], dec_to_round)
             mu_uncert = round(perr[1], dec_to_round)
             std = round(popt[2], dec_to_round)
             std_uncert = round(perr[2], dec_to_round)

             plt.text(0.6, 0.6, '{} \n\n Fit \n$\mu$ = {} $\pm$ {} {} \n$\sigma$ = {} $\pm$ {} {} \n Num \n $\mu$ = {:.5f} {} \n Quant. = {:.5f} {}'.format(extra_string, mu, mu_uncert, unit, std, std_uncert, unit, round(numerical_mean,dec_to_round), unit, round(quantile, dec_to_round), unit), fontsize=9, transform = plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

             plt.xlim(-xlimit, xlimit)

             plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
             if "energies" in target or "momentum" in target:
                plt.xlabel('(true - predicted)/true ')
             else:
                plt.xlabel('true - predicted ' + unit)
             plt.ylabel('count')
             plt.yscale('log')
             plt.ylim((0.1, 200000))
             plt.savefig(f"outputs/pred_{target}_{vertex_axis[i]}_{extra_string}.png")


             residual_lst.append(residuals)
             residual_lst_wcut.append(residuals_cut)

     if False and combine:
         residuals = np.array(residual_lst)
         labels = ['X', 'Y', 'Z']
         if "momentum" in target:
             labels = ['Global']
         figure = corner(residuals.T, bins=50,  labels=labels, range=[(-cut,cut), (-cut,cut), (-cut,cut)]) 
         plt.show()


def fitqun_regression_results(hy_path = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/',
                               npy_path = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/20092023-101855/', target='positions'):
     '''
     Plot fiTQun specific regression results using un_normalize(), regression_analysis(), and read_fitqun_file().
     Args:
         hy_path (str, optional): directory where fitqun_combine.hy and combine_combine.hy files are located. 
         true_path (str, optional): directory where true_positions.npy file is located.
     Returns:
         None
     '''
     # get values out of fitqun file, where mu_1rpos and e_1rpos are the positions of muons and electrons respectively
     (_, labels, _, fitqun_hash), (mu_1rpos, e_1rpos, mu_1rdir, e_1rdir, mu_1rmom, e_1rmom) = fq.read_fitqun_file(hy_path+'fitqun_combine.hy', regression=True)

     print('original mu_1rpos.shape =', mu_1rpos.shape)
     print('original e_1rpos.shape =', e_1rpos.shape)
     # read in the indices file
     idx = np.array(sorted(np.load(npy_path + "/indices.npy")))
     print(f"IDX: {idx.shape}")

     # read in the main HDF5 file that has the rootfiles and event_ids
     hy = h5py.File(hy_path+'combine_combine.hy', "r")
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


     # calculate number of hits 
     events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
     nhits = (events_hits_index[idx+1] - events_hits_index[idx]).squeeze()
     print(nhits)
     nhits_cut=200


     #Apply cuts
     print("WARNING: ONLY LOOKING AT ELECTRON EVENTS")
     event_ids = event_ids[(nhits> nhits_cut)]
     print(f'len roofiles before cut: {rootfiles.shape}')
     rootfiles = rootfiles[(nhits> nhits_cut)]
     print(f'len roofiles after cut: {rootfiles.shape}')
     positions = positions[(nhits> nhits_cut)]
     directions = directions[(nhits> nhits_cut)]
     momenta = momenta[(nhits> nhits_cut)]

     ml_hash = fq.get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)

     print('LEN1 :', positions.shape)
     print('LEN2 :', mu_1rpos.shape)

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
     print(f'intersect: {intersect}, comm1: {comm1.shape}, comm2: {comm2.shape}')
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

     if "positions" in target:
         truth = positions
         fitqun_mu = fitqun_mu_1rpos
         fitqun_e = fitqun_e_1rpos
     elif "directions" in target:
        truth = directions
        fitqun_mu = fitqun_mu_1rdir
        fitqun_e = fitqun_e_1rdir
     elif "momentum" in target:
        truth = momenta
        fitqun_mu = fitqun_mu_1rmom
        fitqun_e = fitqun_e_1rmom

     true_0, pred_0 = [], []
     true_1, pred_1 = [], []
     for i in range(len(fitqun_labels)):
         # LABEL 0 - muons  
         if fitqun_labels[i] == 0:
             true_0.append(truth[i])
             pred_0.append(fitqun_mu[i])

         # LABEL 1 - electrons  
         else:
             true_1.append(truth[i])
             pred_1.append(fitqun_e[i])

     # convert lists to arrayss
     true_0 = np.array(true_0)
     true_1 = np.array(true_1)
     pred_0 = np.array(pred_0)
     pred_1 = np.array(pred_1)

     print('######## MUON EVENTS ########')
     print(true_0.shape)
     regression_analysis(from_path=False, true=true_0, pred=pred_0, target = target, extra_string="fitqun_Muons")

     print('######## ELECTRON EVENTS ########')
     print(true_1.shape)
     regression_analysis(from_path=False, true=true_1, pred=pred_1, target = target, extra_string="fitqun_Electrons")

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


