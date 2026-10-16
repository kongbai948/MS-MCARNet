import os
import numpy as np
import scipy.io as sio


from feature_3D import data_1Dto2D





SEED_TO_DEAP_32_IDX = [
    3, 0, 7, 5, 15, 23, 25, 33, 43, 41, 58, 52, 17, 26, 35, 44,  
    2, 4, 11, 13, 21, 31, 29, 39, 47, 49, 60, 54, 19, 28, 37, 46  
]


def read_file(path: str):
    mat = sio.loadmat(path)
    if 'data' not in mat or 'label' not in mat:
        raise KeyError(f"Missing keys in {path}. Expect 'data' and 'label'.")
    data = np.asarray(mat['data'])
    label = np.asarray(mat['label'])
    label = np.rint(label).astype(np.int64)
    if label.ndim != 2 or label.shape != (15, 185):
        raise ValueError(f"SEED labels must have shape (15, 185), got {label.shape}")
    if not set(np.unique(label).tolist()).issubset({0, 1, 2}):
        raise ValueError("SEED labels must use categorical codes 0=negative, 1=neutral, 2=positive")
    return data, label


def pre_process(path_1d_mat: str):
    
    trial_data, label = read_file(path_1d_mat)
    num_samples = trial_data.shape[0]  

    
    
    data = trial_data.astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    data = (data - mean) / (std + 1e-8)

    
    
    data_3D_output = np.zeros((num_samples, 4, 8, 9), dtype=np.float32)

    sub_vector_len = 62  

    for i in range(num_samples):
        sample_vector = data[i]
        for band in range(4):
            
            band_vec_62 = sample_vector[band * sub_vector_len: (band + 1) * sub_vector_len]

            
            band_vec_32 = band_vec_62[SEED_TO_DEAP_32_IDX]

            
            data_3D_output[i, band, :, :] = data_1Dto2D(band_vec_32)

    
    
    data_3D_reshaped = data_3D_output.reshape(num_samples, 4, 72)
    
    data_3D_final = data_3D_reshaped.reshape(15, 185, 4, 72)

    return data_3D_final, label


if __name__ == '__main__':
    dataset_dir = r".\data_result\SEED_DE_1D"
    result_dir = r".\data_result\SEED_DE_3D_Aligned"

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith('.mat')])
    for f in files:
        print(f"Aligning and processing: {f} ...")
        in_path = os.path.join(dataset_dir, f)

        
        data_3d, label_3d = pre_process(in_path)

        sio.savemat(os.path.join(result_dir, f), {
            "data": data_3d,
            "label": label_3d,
        })
    print("Mission Accomplished! 所有电极已物理对齐。")
