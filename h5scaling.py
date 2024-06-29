import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, MaxAbsScaler

import time

import torch

import sys

# torch.cuda.set_device(7)


from joblib import dump, load


print(sys.path)
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/dataset/cnn/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/dataset/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/watchmal/')
sys.path.append('/home/thoriba/t2k2/t2k_ml_training/WatChMaL/')
from cnn_dataset import CNNDataset, CNNDatasetDeadPMT, CNNDatasetScale


h5file = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/combine_combine.hy'
pmt_pos = '/data/thoriba/t2k/imagefile/skdetsim_imagefile.npy'

cs = {'fitted_scaler': None,
      'scaler_type': 'minmax',
      'sample_size': 10000,
      'dataset_index_file': None
      }
# cnn = CNNDatasetScale(h5file, pmt_pos, channel_scaler=cs)

# dump(cnn.scaler, 'scaler.joblib')

scaler = load('scaler.joblib')

print(scaler.get_params())

test_array = np.array([100*i for i in range(5, 15)])
print('before', test_array)
test_array = scaler.transform(test_array.reshape(-1, 1))

print('after', test_array)



# device = config.gpu_list[rank]




# scaler = StandardScaler(copy=False)
# scaler = QuantileTransformer(copy=False, subsample=1000)
# # scaler = PowerTransformer(copy=False)

# start_time = time.time()

# a = np.load('/data/thoriba/t2k/indices/apr3_eMuPiPlus_1500MeV_small_1/train_val_test_gt200Hits_FCTEST_nFolds10_fold0.npz')

# print(a['train_idxs'])
# print(a['train_idxs'].shape)
# print(a['val_idxs'].shape)
# print(a['test_idxs'].shape)

# print('sum', a['train_idxs'].shape[0] + a['val_idxs'].shape[0] +a['test_idxs'].shape[0] )



# # with np.load('/data/fcormier/t2k/Public/oct20_eMuPosPion_0dwallCut_flat_1/fq_test_idx.npz') as idx:
# #     print(idx['test_idx'])

# with h5py.File('/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/combine_combine.hy', mode='r') as h5all:
#     times = np.ravel(h5all['hit_time'])
#     event_hits_index = np.ravel(h5all['event_hits_index'])

#     # print('len times', len(times))



#     def print_name(name):
#         print(name)

#     h5all.visit(print_name)

#     event_hit_idxs = np.ravel(h5all['event_hits_index'])
#     print(event_hit_idxs)
#     print(event_hit_idxs.shape)

# #     # length = len(times)
# #     # hi = 0
# #     # while hi < length:
# #     #     last = hi + 100000
# #     #     if last > length:
# #     #         last = length
# #     #     scaler.partial_fit(times.reshape(-1, 1)[hi:last,:])

# #     #     hi = hi + 100000

# #     #     print('partial fitting', hi)

#     # takes forever (without GPU)
#     scaler.fit(times.reshape(-1, 1))
#     # scaler.fit(times.reshape(-1, 1)[1000:2000,:])
#     print('fit done')
#     print(scaler.get_params())
#     # print(scaler.quantiles_)
#     # scaler.transform(times.reshape(-1, 1))
    
# end_time = time.time()
# print('done. took', end_time - start_time)

    