

import scipy.io as sio
from train_model import *


class PrepareData:
    def __init__(self, args):
        
        
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
    def _entropy_prefix(self):
        entropy = str(getattr(self.args, 'entropy', 'DE')).strip().lower()
        if entropy in {'de', 'differential_entropy'}:
            return 'DE'
        if entropy in {'spen', 'spectral_entropy'}:
            return 'SpEn'
        if entropy in {'peen', 'permutation_entropy'}:
            return 'PeEn'
        if entropy in {'fuzzyen', 'fuzzy_entropy'}:
            return 'FuzzyEn'
        raise ValueError(f'Unsupported entropy: {getattr(self.args, "entropy", None)}')

    def run(self, subject_list, split=False, expand=False, feature=False):
        
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            
            if expand:
                
                data_ = np.expand_dims(data_, axis=-3)
            print('Data and label prepared for sub{}!'.format(sub))
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    def load_data_per_subject(self, sub):
        
        sub += 1
        prefix = self._entropy_prefix()
        if (sub < 10):
            sub_code = str(prefix + '_s0' + str(sub) + '.mat')
        else:
            sub_code = str(prefix + '_s' + str(sub) + '.mat')
        subject_path = os.path.join(self.data_path, sub_code)
        subject = sio.loadmat(subject_path)
        if self.label_type == 'A':
            label = subject['arousal_labels']
        elif self.label_type == 'V':
            label = subject['valence_labels']
        data = subject['data']  
        return data, label

    def save(self, data, label, sub):
        
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

