import os
import re
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

STANDARD_SEED_TRIAL_LABELS = np.array(
    [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1],
    dtype=np.int64,
)
SEED_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}


def encode_seed_labels(labels):
    labels = np.asarray(labels).reshape(-1)
    if not np.all(np.isfinite(labels)) or not np.allclose(labels, np.rint(labels)):
        raise ValueError("SEED emotion labels must be finite integer category codes")
    labels = np.rint(labels).astype(np.int64)
    unique_labels = set(np.unique(labels).tolist())
    if unique_labels == {-1, 0, 1}:
        return np.array([SEED_LABEL_TO_CLASS[int(label)] for label in labels], dtype=np.int64)
    if unique_labels == {0, 1, 2}:
        return labels
    raise ValueError(f"Expected SEED emotion labels {-1, 0, 1} or class IDs {0, 1, 2}, got {sorted(unique_labels)}")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    if variance <= 0: return 0.0
    return math.log(2 * math.pi * math.e * variance) / 2


def load_seed_labels(label_mat_path=None) -> np.ndarray:
    if label_mat_path is not None and os.path.isfile(label_mat_path):
        mat = sio.loadmat(label_mat_path)
        if 'label' not in mat:
            raise KeyError(f"'label' not found in {label_mat_path}")
        labels = np.asarray(mat['label']).squeeze()
    else:
        labels = STANDARD_SEED_TRIAL_LABELS.copy()

    if labels.size != 15:
        raise ValueError(f"Expect 15 SEED trial labels, got shape={labels.shape} size={labels.size}")
    return encode_seed_labels(labels)


def decompose_seed(file_path, segments_per_trial=185):
    
    data_mat = sio.loadmat(file_path)
    
    keys = [k for k in data_mat.keys() if 'eeg' in k.lower()]
    keys.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    num_channels = 62
    frequency = 200  
    samples_per_seg = 200  
    samples_per_trial = segments_per_trial * samples_per_seg  

    
    
    decomposed_de = np.empty([0, 4, 62])

    if len(keys) != 15:
        raise ValueError(f"Expect 15 eeg trials in {os.path.basename(file_path)}, got {len(keys)} keys={keys}")

    for key in keys:
        trial_data = data_mat[key]
        if trial_data.ndim != 2 or trial_data.shape[0] != num_channels:
            raise ValueError(f"Unexpected EEG shape for {key}: {trial_data.shape}, expected (62, samples)")
        if trial_data.shape[1] < samples_per_trial:
            raise ValueError(f"Trial {key} has {trial_data.shape[1]} samples; at least {samples_per_trial} are required")
        trial_data = trial_data[:, :samples_per_trial]

        
        
        temp_de = np.empty([0, segments_per_trial])

        for channel in range(num_channels):
            channel_signal = trial_data[channel, :samples_per_trial]

            
            theta = butter_bandpass_filter(channel_signal, 4, 8, frequency)
            alpha = butter_bandpass_filter(channel_signal, 8, 14, frequency)
            beta = butter_bandpass_filter(channel_signal, 14, 31, frequency)
            gamma = butter_bandpass_filter(channel_signal, 31, 45, frequency)

            DE_theta = []
            DE_alpha = []
            DE_beta = []
            DE_gamma = []

            for index in range(segments_per_trial):
                start = index * samples_per_seg
                end = (index + 1) * samples_per_seg
                
                DE_theta.append(compute_DE(theta[start:end]))
                DE_alpha.append(compute_DE(alpha[start:end]))
                DE_beta.append(compute_DE(beta[start:end]))
                DE_gamma.append(compute_DE(gamma[start:end]))

            
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

        
        
        temp_trial_de = temp_de.reshape(-1, 4, segments_per_trial).transpose([2, 1, 0])
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

    
    
    total_de_flat = decomposed_de.reshape(-1, 248)

    return total_de_flat


def get_seed_labels(labels_15, segments_per_trial=185):
    labels = encode_seed_labels(labels_15)
    if labels.size != 15:
        raise ValueError(f"Expect 15 trial labels, got {labels.size}")
    return np.repeat(labels[:, None], segments_per_trial, axis=1)


if __name__ == '__main__':
    dataset_dir = r".\data"
    result_dir = r".\data_result\SEED_DE_1D"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    
    label_path = os.path.join(dataset_dir, 'label.mat')
    labels_15 = load_seed_labels(label_path)
    eeg_files = [
        file for file in os.listdir(dataset_dir)
        if file.lower().endswith('.mat') and file.lower() != 'label.mat'
    ]
    if not eeg_files:
        raise FileNotFoundError(f"No SEED EEG .mat files found under {dataset_dir}")
    if os.path.isfile(label_path):
        print(f"Using categorical SEED labels from {label_path}")
    else:
        print(
            "label.mat not found; using the standard SEED 15-trial "
            "categorical label sequence"
        )

    for file in eeg_files:

        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)

        
        
        data_flat = decompose_seed(file_path, segments_per_trial=185)

        
        label_2d = get_seed_labels(labels_15, segments_per_trial=185)
        if data_flat.shape[0] != 15 * 185 or data_flat.shape[1] != 248:
            raise ValueError(f"Unexpected feature shape for {file}: {data_flat.shape}, expect (2775,248)")

        
        sio.savemat(os.path.join(result_dir, "DE_" + file), {
            "data": data_flat,
            "label": label_2d,
            "base_data": np.zeros((15, 248), dtype=np.float32)  
        })
