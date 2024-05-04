import numpy as np
import h5py

data = h5py.File("/fast_scratch/fcormier/t2k/ml/skdetsim/oct20_eMuPosPion_0dwallCut_flat_1/combine_combine.hy")
data_rootfiles = np.array(data['root_files'])

data_rootfiles = [x.decode('UTF-8') for x in data_rootfiles]
data_rootfiles = np.char.split(data_rootfiles,'/')
data_rootfiles = [x[-1] for x in data_rootfiles]
data_rootfiles = np.char.split(data_rootfiles,'_')
data_rootfiles = [x[1] for x in data_rootfiles]
data_rootfiles = np.char.split(data_rootfiles,'.')
data_rootfiles = [x[0] for x in data_rootfiles]

print("data rootfiles analyzed")

fitqun=h5py.File("/fast_scratch/fcormier//t2k/ml/training/resnet_nov7_eMuPosPions_0dwall_1M_flat_1/06112023-205208/fitqun_combine.hy","r")
unique_fq_rootfiles = np.unique(np.array(fitqun['root_files']))

unique_fq_rootfiles = [x.decode('UTF-8') for x in unique_fq_rootfiles]
unique_fq_rootfiles = np.char.split(unique_fq_rootfiles,'/')
unique_fq_rootfiles = [x[-1] for x in unique_fq_rootfiles]
unique_fq_rootfiles = np.char.split(unique_fq_rootfiles,'_')
unique_fq_rootfiles = [x[1] for x in unique_fq_rootfiles]


print("fq rootfiles analyzed")

new_idx = np.nonzero(np.array(unique_fq_rootfiles)[:,None] == np.array(data_rootfiles))[1]

print(new_idx)

np.savez('/fast_scratch/fcormier//t2k/ml/training/resnet_nov7_eMuPosPions_0dwall_1M_flat_1/fq_test_idx.npz',
        test_idxs=new_idx)