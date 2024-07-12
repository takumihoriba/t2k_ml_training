import csv
import sys
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def output_column_from_csv(filename, col_name):

    x = []
    col_val = []

    x_best = []
    col_val_best = []

    with open (filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                for j, name in enumerate(row):
                    if col_name in row[j]:
                        col_idx = j
                    if "iteration" in row[j]:
                        x_idx = j
                    if "saved_best" in row[j]:
                        best_idx = j
            if i > 0:
                x.append(int(row[x_idx]))
                col_val.append(float(row[col_idx]))
                row_is_best = False
                if "val" in filename:
                    if "True" in row[best_idx]:
                        row_is_best = True
                if "val" in filename and row_is_best:
                    x_best.append(int(row[x_idx]))
                    col_val_best.append(float(row[col_idx]))

    return x, col_val, x_best, col_val_best

#First argument is where to save plot
output_path = sys.argv[1]
files = sys.argv[2:]


colors = ['red', 'blue', 'green', 'black', 'pink']
markers = ['.','^','o']

plt.figure(figsize=(10, 5))

for i, file in enumerate(files):
    x_temp, y_temp, x_best_temp, y_best_temp = output_column_from_csv(file, "loss")
    base_filename = os.path.basename(os.path.normpath(file))
    plt.scatter(x_temp, y_temp, c=colors[i], marker=markers[i], label=base_filename, alpha=0.5)
    if len(x_best_temp) > 0:
        plt.scatter(x_best_temp, y_best_temp, c='orange', marker=markers[i], label='Best')
    print(f'Min loss: {np.amin(y_temp)} for file {base_filename}')

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.ylim(0.1,5)
plt.yscale('log')
plt.legend()
plt.savefig(output_path)
plt.show()



#/fast_scratch/fcormier/t2k/ml/training/resnet_feb6_watchmalMerge_regOnly_electronOnly_0dwall_5M_flat_1/06022024-153735
