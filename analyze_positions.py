import numpy as np

import sys

import matplotlib
from matplotlib import pyplot as plt

#First argument is where to save plot
output_path = sys.argv[1]
#Second one where to get data
files = sys.argv[2]

preds = np.load(files+'predicted_positions.npy')
truth = np.load(files+'positions.npy')

#print(preds[:,0].shape)
#print(truth[:,0].shape)

pred_x = preds[:,0]*1600 
pred_y = preds[:,1]*1600
pred_z = preds[:,2]*1600 

truth_x = truth[:,0]*1600 
truth_y = truth[:,1]*1600 
truth_z = truth[:,2]*1600 

diff_x = pred_x - truth_x
#diff_x = diff_x[np.abs(diff_x) < 1000]
diff_y = pred_y - truth_y
#diff_y = diff_y[np.abs(diff_y) < 1000]
diff_z = pred_z - truth_z
#diff_z = diff_z[np.abs(diff_z) < 1000]


print(f"X mean: {np.mean(diff_x)}, std: {np.std(diff_x)}, max diff: {np.amax(diff_x)}")
print(f"Y mean: {np.mean(diff_y)}, std: {np.std(diff_y)}, max diff: {np.amax(diff_y)}")
print(f"Z mean: {np.mean(diff_z)}, std: {np.std(diff_z)}, max diff: {np.amax(diff_z)}")

plt.hist(diff_x, range=[-2000,2000], bins=50)
plt.xlabel("Pred X - True X [cm]")
plt.yscale('log')
plt.ylabel("Counts")
plt.show()