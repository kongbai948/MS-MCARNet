import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, welch
import argparse


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


def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    
    return data


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def _safe_std(x):
    x = np.asarray(x)
    return float(np.std(x)) if x.size else 0.0


def compute_SpEn(signal, sf=128):
    
    x = np.asarray(signal, dtype=float)
    N = len(x)

    
    if _safe_std(x) < 1e-8:
        return 0.0

    
    
    psd = np.abs(np.fft.rfft(x)) ** 2 / N

    
    psd_sum = np.sum(psd)
    if psd_sum <= 0:
        return 0.0

    p = psd / psd_sum

    
    
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0

    h = -np.sum(p * np.log(p))

    
    
    return h / np.log(len(p))


def compute_PeEn(signal, order=3, delay=1, normalize=True):
    
    x = np.asarray(signal, dtype=float)
    if _safe_std(x) < 1e-8:
        return 0.0
    n = len(x)
    order = int(order)
    delay = int(delay)
    if order < 2 or delay < 1:
        raise ValueError("order must be >=2 and delay must be >=1")
    emb_len = n - (order - 1) * delay
    if emb_len <= 1:
        return 0.0
    patterns = {}
    for i in range(emb_len):
        window = x[i:(i + order * delay):delay]
        if window.size != order:
            continue
        key = tuple(np.argsort(window, kind="mergesort"))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    if counts.size <= 1:
        return 0.0
    p = counts / float(np.sum(counts))
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p)))
    if not normalize:
        return h
    return h / float(np.log(math.factorial(order)))


def compute_FuzzyEn(signal, order=2, r=None, n=2):
    
    x = np.asarray(signal, dtype=float)
    if _safe_std(x) < 1e-8:
        return 0.0
    m = int(order)
    n = int(n)
    if m < 1 or n < 1:
        raise ValueError("order and n must be >= 1")
    if r is None:
        r = 0.2 * float(np.std(x))
    r = float(r)
    if r <= 0:
        return 0.0

    def _phi(mm):
        N = len(x)
        L = N - mm + 1
        if L <= 1:
            return 0.0
        X = np.array([x[i:i + mm] for i in range(L)], dtype=float)
        X = X - np.mean(X, axis=1, keepdims=True)  
        
        dist = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        np.fill_diagonal(dist, np.inf)
        D = np.exp(-(dist ** n) / r)
        
        return float(np.sum(D)) / float(L * (L - 1))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 1e-12 or phi_m1 <= 1e-12:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def _get_entropy_fn(entropy_type: str):
    et = (entropy_type or "DE").strip().lower()
    if et in {"de", "differential_entropy"}:
        return "DE", compute_DE
    if et in {"spen", "spectral_entropy"}:
        return "SpEn", lambda x: compute_SpEn(x, sf=128)
    if et in {"peen", "permutation_entropy"}:
        return "PeEn", lambda x: compute_PeEn(x, order=3, delay=1, normalize=True)
    if et in {"fuzzyen", "fuzzy_entropy"}:
        return "FuzzyEn", lambda x: compute_FuzzyEn(x, order=2, r=0.2 * float(np.std(x)) if _safe_std(x) >= 1e-8 else 0.0)
    raise ValueError(f"Unsupported entropy_type: {entropy_type}")


def decompose(file, window_s=0.5, entropy_type="DE"):
    
    start_index = 384  
    data = read_file(file)
    shape = data.shape
    frequency = 128
    window_s = float(window_s)
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    win = int(round(window_s * frequency))
    if win <= 0:
        raise ValueError("window_s too small for current sampling-rate")
    trial_len = shape[2] - start_index
    num_segments = trial_len // win
    if num_segments <= 0:
        raise ValueError("window_s too large for trial length")
    base_len = start_index
    base_segments = base_len // win
    if base_segments <= 0:
        raise ValueError("window_s too large for base length (3s=384 points)")

    entropy_name, entropy_fn = _get_entropy_fn(entropy_type)

    decomposed_de = np.empty([0, 4, num_segments])

    base_DE = np.empty([0, 128])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])

        temp_de = np.empty([0, num_segments])

        for channel in range(32):
            trial_signal = data[trial, channel, 384:]
            base_signal = data[trial, channel, :384]
            
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            def _base_avg(x):
                total = 0.0
                for s in range(base_segments):
                    seg = x[s * win:(s + 1) * win]
                    total += float(entropy_fn(seg))
                return total / float(base_segments)

            base_theta_DE = _base_avg(base_theta)
            base_alpha_DE = _base_avg(base_alpha)
            base_beta_DE = _base_avg(base_beta)
            base_gamma_DE = _base_avg(base_gamma)

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(num_segments):
                st = index * win
                ed = (index + 1) * win
                DE_theta = np.append(DE_theta, float(entropy_fn(theta[st:ed])) - base_theta_DE)
                DE_alpha = np.append(DE_alpha, float(entropy_fn(alpha[st:ed])) - base_alpha_DE)
                DE_beta = np.append(DE_beta, float(entropy_fn(beta[st:ed])) - base_beta_DE)
                DE_gamma = np.append(DE_gamma, float(entropy_fn(gamma[st:ed])) - base_gamma_DE)
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])
        temp_trial_de = temp_de.reshape(-1, 4, num_segments)
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)
        base_DE = np.vstack([base_DE, temp_base_DE])
    decomposed_de = decomposed_de.reshape(-1, 32, 4, num_segments).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)

    print("base_DE shape:", base_DE.shape)
    print(f"trial_{entropy_name} shape:", decomposed_de.shape)
    return base_DE, decomposed_de


def get_labels(file, num_segments=120):
    
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    for i in range(len(valence_labels)):
        for j in range(0, int(num_segments)):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("labels:", final_arousal_labels.shape)
    final_arousal_labels = final_arousal_labels.reshape(40, -1)
    final_valence_labels = final_valence_labels.reshape(40, -1)
    return final_arousal_labels, final_valence_labels


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r".\data\matlabPREPROCESSED/")
    parser.add_argument('--result-dir', type=str, default=None,
                        help=r'输出目录。默认按 window/entropy 自动命名到 .\data_result\with_base_{window}[_entropy]\ ')
    parser.add_argument('--window', type=float, default=0.5, choices=[0.5, 1.0, 2.0])
    parser.add_argument('--entropy', type=str, default='DE',
                        help='DE/SpEn/PeEn/FuzzyEn (不区分大小写)')
    args = parser.parse_args()

    window_tag = str(args.window).rstrip('0').rstrip('.')  
    entropy_tag, _ = _get_entropy_fn(args.entropy)
    entropy_suffix = '' if entropy_tag == 'DE' else f'_{entropy_tag.lower()}'
    result_dir = args.result_dir or rf".\data_result\with_base_{window_tag}{entropy_suffix}"
    if os.path.isdir(result_dir) is False:
        os.makedirs(result_dir)

    frequency = 128
    win = int(round(float(args.window) * frequency))
    num_segments = (8064 - 384) // win  

    for file in os.listdir(args.dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(args.dataset_dir, file)
        base_feat, trial_feat = decompose(file_path, window_s=args.window, entropy_type=args.entropy)
        arousal_labels, valence_labels = get_labels(file_path, num_segments=num_segments)
        sio.savemat(os.path.join(result_dir, f"{entropy_tag}_{file}"),
                    {"base_data": base_feat, "data": trial_feat, "valence_labels": valence_labels,
                     "arousal_labels": arousal_labels})
