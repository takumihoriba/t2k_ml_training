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

# h5file = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/combine_combine.hy'
# h5file = '/data/thoriba/t2k/datasets/apr3_eMuPiPlus_1500MeV_small_1/multi_combine.hy'
pmt_pos = '/data/thoriba/t2k/imagefile/skdetsim_imagefile.npy'

# bigger dataset and indices taken from reg (elec_pos)
h5file = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/combine_combine.hy'
indices = '/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/train_val_test_nFolds3_fold0.npz'


# ['/home/thoriba/t2k2/t2k_ml_training/chain_scaler_1_quantile_normal.joblib', '/home/thoriba/t2k2/t2k_ml_training/chain_scaler_2_minmax.joblib']

cs = {'fitted_scaler': None, # set None to fit a new scaler, if provided, only this and transform_per_batch will be used.
      'scaler_type': 'minmax',
      'sample_size': 10000,
      'dataset_index_file': indices,
      'transform_per_batch': False
      }
cnn = CNNDatasetScale(h5file, pmt_pos, channel_scaler=cs)

# gotten_item = cnn.__getitem__(100)

for i in range(10):
      cnn.__getitem__(i)

# print(gotten_item.keys())

# print(gotten_item['data'])

# dump(cnn.scaler, 'scaler_minmax_test.joblib')

# scaler = load('scaler_minmax_test.joblib')
# scaler1 = load('chain_scaler_1_quantile_normal.joblib')
# scaler2 = load('chain_scaler_2_minmax.joblib')

# print(scaler1.get_params())
# print(scaler2.get_params())



# test_array = np.array([100*i for i in range(5, 15)])
# print('before', test_array)
# test_array = scaler1.transform(test_array.reshape(-1, 1))

# print('after', test_array)

# test_array = scaler1.transform(test_array.reshape(-1, 1))

# with h5py.File('/fast_scratch_2/fcormier/t2k/ml/data/oct20_combine_flatE/combine_combine.hy', mode='r') as h5all:
#     times = np.ravel(h5all['hit_time'])
#     event_hits_index = np.ravel(h5all['event_hits_index'])




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

    