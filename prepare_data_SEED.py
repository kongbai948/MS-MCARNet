

import os
import os.path as osp
import h5py
import numpy as np
import scipy.io as sio


class PrepareData:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path

    def run(self, subject_list, split=False, expand=False, feature=False):
        
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            if expand:
                
                data_ = np.expand_dims(data_, axis=-3)
            print('Data and label prepared for sub{}!'.format(sub))
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    def _list_subject_mats(self):
        mats = sorted([f for f in os.listdir(self.data_path) if f.lower().endswith('.mat')])
        if len(mats) < 1:
            raise FileNotFoundError(f"No .mat files found under {self.data_path}")
        return mats

    def load_data_per_subject(self, sub: int):
        mats = self._list_subject_mats()
        if sub < 0 or sub >= len(mats):
            raise IndexError(f"sub={sub} out of range for {len(mats)} mat files")
        mat_path = osp.join(self.data_path, mats[sub])
        m = sio.loadmat(mat_path)
        if 'data' not in m or 'label' not in m:
            raise KeyError(f"Missing 'data'/'label' in {mat_path}. keys={list(m.keys())}")
        data = np.asarray(m['data'])
        label = np.asarray(m['label'])
        if data.shape != (15, 185, 4, 72):
            raise ValueError(f"Expect data shape (15,185,4,72), got {data.shape} in {mat_path}")
        if label.shape != (15, 185):
            raise ValueError(f"Expect label shape (15,185), got {label.shape} in {mat_path}")
        label = np.rint(label).astype(np.int64)
        if not set(np.unique(label).tolist()).issubset({0, 1, 2}):
            raise ValueError(
                f"SEED labels must use categorical codes 0=negative, 1=neutral, 2=positive; "
                f"got {sorted(np.unique(label).tolist())} in {mat_path}"
            )
        return data, label

    def save(self, data, label, sub):
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()
