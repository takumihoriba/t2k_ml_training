import numpy as np

import sys

import matplotlib
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

from plotting import regression_analysis

import WatChMaL.analysis.utils.fitqun as fq

import h5py

def gaussian(x, a, mean, sigma):
     return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

#First argument is where to save plot
output_path = sys.argv[1]
#Second one where to get data
files = sys.argv[2]
#Third argument the target variable
target = str(sys.argv[3])
#Fourth argument if you want to only keep indices that match fitqun file
preds = np.load(files+'predicted_'+target+'.npy')
truth = np.load(files+target+'.npy')
labels = np.load(files + 'labels.npy')



if len(sys.argv) > 4:
     fitqun_path = sys.argv[4]
     (_, fq_labels, _, fitqun_hash), (mu_1rpos, e_1rpos, mu_1rdir, e_1rdir, mu_1rmom, e_1rmom) = fq.read_fitqun_file(fitqun_path+'fitqun_combine.hy', regression=True)
     ml_combine_path = sys.argv[5]
     hy = h5py.File(ml_combine_path+'combine_combine.hy', "r")
     indices = np.load(files + 'indices.npy')

     # calculate number of hits 
     events_hits_index = np.append(hy['event_hits_index'], hy['hit_pmt'].shape[0])
     nhits = (events_hits_index[indices+1] - events_hits_index[indices]).squeeze()
     print(nhits)
     print(f"IDX: {indices.shape}")
     rootfiles = np.array(hy['root_files'])[indices].squeeze()
     event_ids = np.array(hy['event_ids'])[indices].squeeze()

     nhits_cut = 200

     #Apply cuts
     print("WARNING: ONLY LOOKING AT ELECTRON EVENTS")
     event_ids = event_ids[(nhits> nhits_cut)]
     print(f'len roofiles before cut: {rootfiles.shape}')
     rootfiles = rootfiles[(nhits> nhits_cut)]
     print(f'len roofiles after cut: {rootfiles.shape}')
     preds = preds[(nhits> nhits_cut)]
     truth = truth[(nhits> nhits_cut)]
     labels = labels[(nhits> nhits_cut)]


     ml_hash = fq.get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=False)
     intersect, comm1, comm2 = np.intersect1d(fitqun_hash, ml_hash, return_indices=True)
     print(f'intersect: {intersect}, comm1: {comm1.shape}, comm2: {comm2.shape}')
     preds = preds[comm2]
     truth = truth[comm2]
     labels = labels[comm2]

preds = preds[labels==1]
truth = truth[labels==1]

#print(preds[:,0].shape)
#print(truth[:,0].shape)

correction = 1
pred_x = preds[:,0]*correction 
pred_y = preds[:,1]*correction
pred_z = preds[:,2]*correction 
if "positions" in target:
     correction = 1600

if "positions" in target or "directions" in target:

     truth_x = truth[:,0]*correction 
     truth_y = truth[:,1]*correction 
     truth_z = truth[:,2]*correction 
     truth_0 = np.stack((truth_x, truth_y, truth_z), axis=1)
     print(truth_0.shape)
     pred_0 = np.stack((pred_x, pred_y, pred_z), axis=1)
     print(pred_0.shape)
if "energies" in target:
     truth_0 = np.ravel(truth)
     print(truth_0.shape)
     pred_0 = np.ravel(preds)
     print(pred_0.shape)



regression_analysis(from_path=False, true=truth_0, pred=pred_0, target=target, extra_string="ML_Electrons")
exit

diff_x = pred_x - truth_x
#diff_x = diff_x[np.abs(diff_x) < 1000]
diff_y = pred_y - truth_y
#diff_y = diff_y[np.abs(diff_y) < 1000]
diff_z = pred_z - truth_z
#diff_z = diff_z[np.abs(diff_z) < 1000]


print(f"X mean: {np.mean(diff_x)}, std: {np.std(diff_x)}, max diff: {np.amax(diff_x)}")
print(f"Y mean: {np.mean(diff_y)}, std: {np.std(diff_y)}, max diff: {np.amax(diff_y)}")
print(f"Z mean: {np.mean(diff_z)}, std: {np.std(diff_z)}, max diff: {np.amax(diff_z)}")

line = np.linspace(-1600, 1600, 10000) 

yhist, xhist, _ = plt.hist(diff_x, bins=100, alpha=0.7, color='green')
popt, pcov = curve_fit(gaussian, (xhist[1:]+xhist[:-1])/2, yhist, bounds=(-np.inf, np.inf), p0=[40, 0, 70])    
perr = np.sqrt(np.diag(pcov))
# round numbers
mu = round(popt[1], 2)
mu_uncert = round(perr[1], 2)
std = round(popt[2], 2)
std_uncert = round(perr[2], 2)
print(f"Gaussian std: {std}")

plt.plot(line, gaussian(line, *popt), alpha=1, color='black', label='guassian fit')

#plt.hist(diff_x, range=[-2000,2000], bins=50)
plt.xlabel("Pred X - True X [cm]")
plt.yscale('log')
plt.ylabel("Counts")
plt.ylim((0.1, 200000))
plt.show()
