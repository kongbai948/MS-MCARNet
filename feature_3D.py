from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import os
import argparse


def read_file(file):
    file = sio.loadmat(file)
    trial_data = file['data']
    base_data = file["base_data"]
    return trial_data, base_data, file["arousal_labels"], file["valence_labels"]


def get_vector_deviation(vector1, vector2):
    return vector1 - vector2


def get_dataset_deviation(trial_data, base_data, segments_per_trial):
    new_dataset = np.empty([0, 128])
    total_samples = trial_data.shape[0]
    for i in range(total_samples):
        base_index = i // segments_per_trial
        
        base_index = 39 if base_index == 40 else base_index
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        
        new_dataset = np.vstack([new_dataset, new_record])
    
    return new_dataset


def data_1Dto2D(data, Y=8, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, data[1], data[0], 0, data[16], data[17], 0, 0)
    data_2D[1] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[2] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[3] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[4] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[5] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[6] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[7] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    
    return data_2D


def pre_process(path, y_n):
    
    data_3D = np.empty([0, 8, 9])
    sub_vector_len = 32
    trial_data, base_data, arousal_labels, valence_labels = read_file(path)
    n_trials = int(base_data.shape[0])
    total_samples = int(trial_data.shape[0])
    segments_per_trial = total_samples // n_trials
    if y_n == "yes":
        data = get_dataset_deviation(trial_data, base_data, segments_per_trial)
        data = preprocessing.scale(data, axis=1, with_mean=True, with_std=True, copy=True)
    else:
        data = preprocessing.scale(trial_data, axis=1, with_mean=True, with_std=True, copy=True)
    
    for vector in data:
        for band in range(0, 4):
            data_2D_temp = data_1Dto2D(vector[band * sub_vector_len:(band + 1) * sub_vector_len])
            data_2D_temp = data_2D_temp.reshape(1, 8, 9)
            
            data_3D = np.vstack([data_3D, data_2D_temp])
    data_3D = data_3D.reshape(-1, 4, 8, 9).reshape(total_samples, 4, -1).reshape(n_trials, -1, 4, 72)
    print("final data shape:", data_3D.shape)
    return data_3D, arousal_labels, valence_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=float, default=0.5, choices=[0.5, 1.0, 2.0])
    parser.add_argument('--entropy', type=str, default='DE',
                        help='DE/SpEn/PeEn/FuzzyEn (不区分大小写)')
    parser.add_argument('--use-baseline', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='1D特征输入目录。默认根据 window/entropy 自动推导')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='3D特征输出目录。默认根据 window/entropy/use-baseline 自动推导')
    args = parser.parse_args()

    entropy_key = (args.entropy or "DE").strip().lower()
    if entropy_key in {"de", "differential_entropy"}:
        entropy_suffix = ""
    elif entropy_key in {"spen", "spectral_entropy"}:
        entropy_suffix = "_spen"
    elif entropy_key in {"peen", "permutation_entropy"}:
        entropy_suffix = "_peen"
    elif entropy_key in {"fuzzyen", "fuzzy_entropy"}:
        entropy_suffix = "_fuzzyen"
    else:
        raise ValueError(f"Unsupported --entropy: {args.entropy}")

    window_tag = str(args.window).rstrip('0').rstrip('.')
    dataset_dir = args.dataset_dir or rf".\data_result\with_base_{window_tag}{entropy_suffix}"

    if args.use_baseline == "yes":
        result_dir = args.result_dir or rf".\data_result\with_base_{window_tag}{entropy_suffix}"
    else:
        result_dir = args.result_dir or rf".\data_result\without_base_{window_tag}{entropy_suffix}"
    if os.path.isdir(result_dir) is False:
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        data, arousal_labels, valence_labels = pre_process(file_path, args.use_baseline)
        print("final shape:", data.shape)
        sio.savemat(os.path.join(result_dir, file),
                    {"data": data, "valence_labels": valence_labels, "arousal_labels": arousal_labels})
        
