from cross_validation import *
from prepare_data_SEED import *
import argparse
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='SEED')
    parser.add_argument('--window', type=float, default=1.0, choices=[1.0],
                        help='窗口长度(秒)，当前 SEED pipeline 固定为 1s')
    parser.add_argument('--data-path', type=str,
                        default=r'.\data_result\SEED_DE_3D_Aligned',
                        help='feature_3D_SEED.py 输出目录（包含每个被试的 .mat）')
    parser.add_argument('--subjects', type=int, default=None,
                        help='被试文件数量；不指定时从 --data-path 自动识别')
    parser.add_argument('--num_class', type=int, default=3, choices=[3],
                        help='SEED 为三分类：0/1/2')
    
    parser.add_argument('--label-type', type=str, default='emotion',
                        help='SEED categorical labels: 0=negative, 1=neutral, 2=positive')
    parser.add_argument('--segment', type=float, default=1.0,
                        help='segment length in seconds (固定 1.0)')
    parser.add_argument('--trial-duration', type=float, default=185.0,
                        help='trial duration in seconds for voting (固定 185s)')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=200)
    parser.add_argument('--input-shape', type=tuple, default=(1, 4, 72))
    parser.add_argument('--data-format', type=str, default='raw_1s',
                        help='仅参与输出 hdf 目录命名')

    
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument('--max-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)

    

    
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num-cores', type=int, default=4)
    parser.add_argument('--core-dim', type=int, default=32)
    parser.add_argument('--mcar-layers', type=int, default=2)
    parser.add_argument('--msconv-channels', type=int, default=15)
    parser.add_argument('--msconv-pool', type=int, default=4)
    parser.add_argument('--prepare', action='store_true',
                        help='Force regeneration of HDF5 files from --data-path')

    
    parser.add_argument('--reproduce', type=bool, default=False)

    parser.add_argument('--experiment', type=str, choices=['Idex', 'Dedx'], default='Dedx',
                        help='independent dependence experiment')
    parser.add_argument('--idex-first-fold-only', action='store_true',
                        help='Idex 仅跑第 1 个 fold，跑完该 fold 全部 epoch 后停止')
    args = parser.parse_args()
    args.model = 'MS-MCARNet'

    seed_mat_files = sorted(
        file for file in os.listdir(args.data_path)
        if file.lower().endswith('.mat')
    ) if os.path.isdir(args.data_path) else []
    if not seed_mat_files:
        raise FileNotFoundError(f'No SEED .mat files found under {args.data_path}')
    if args.subjects is None:
        args.subjects = len(seed_mat_files)
    elif args.subjects < 1 or args.subjects > len(seed_mat_files):
        raise ValueError(
            f'--subjects must be between 1 and {len(seed_mat_files)} '
            f'for {args.data_path}, got {args.subjects}'
        )
    sub_to_run = np.arange(args.subjects)

    
    
    prepared_dir = os.path.join(
        os.getcwd(), f'data_{args.data_format}_{args.dataset}_{args.label_type}'
    )
    prepared_files = [os.path.join(prepared_dir, f'sub{sub}.hdf') for sub in sub_to_run]
    if args.prepare or not all(os.path.isfile(path) for path in prepared_files):
        pd = PrepareData(args)
        pd.run(sub_to_run, split=True, feature=False, expand=False)
    else:
        print(f'>>> Reusing prepared SEED HDF5 files under {prepared_dir}')

    if args.experiment == 'Dedx':
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        cv.dependent(subject=sub_to_run, fold=5, reproduce=args.reproduce)
    elif args.experiment == 'Idex':
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        
        
        cv.independence(subject=sub_to_run, fold=args.subjects, reproduce=args.reproduce)
