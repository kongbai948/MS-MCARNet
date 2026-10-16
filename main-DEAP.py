from cross_validation import *
from prepare_data_DEAP import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--window', type=float, default=None, choices=[0.5, 1.0, 2.0],
                        help='窗口长度(秒): 0.5/1/2, 指定后自动设置 data-path/data-format/segment')
    parser.add_argument('--entropy', type=str, default='DE',
                        help='选择读取哪种熵特征目录: DE/SpEn/PeEn/FuzzyEn (不区分大小写)。'
                             '默认 DE 对应 data_result/with_base_{window}；其他熵对应 with_base_{window}_{entropy}')
    parser.add_argument('--data-path', type=str,
                        default=r'.\data_result\with_base_0.5')
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num_class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V'])
    parser.add_argument('--segment', type=float, default=0.5, help='segment length in seconds (0.5/1/2)')
    parser.add_argument('--trial-duration', type=float, default=60.0, help='trial duration in seconds for voting')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1, 28, 512))
    parser.add_argument('--data-format', type=str, default='raw')
    
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument('--max-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
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

    
    parser.add_argument('--reproduce', type=bool, default=False)

    parser.add_argument('--experiment', type=str, choices=['Idex', 'Dedx'], default='Dedx',
                        help='independent dependence experiment')
    parser.add_argument('--idex-first-fold-only', action='store_true',
                        help='Idex 仅跑第 1 个 fold，跑完该 fold 全部 epoch 后停止')
    args = parser.parse_args()
    args.model = 'MS-MCARNet'

    entropy_key = (args.entropy or "DE").strip().lower()
    if entropy_key in {"de", "differential_entropy"}:
        entropy_suffix = ""
        entropy_tag = "DE"
    elif entropy_key in {"spen", "spectral_entropy"}:
        entropy_suffix = "_spen"
        entropy_tag = "SpEn"
    elif entropy_key in {"peen", "permutation_entropy"}:
        entropy_suffix = "_peen"
        entropy_tag = "PeEn"
    elif entropy_key in {"fuzzyen", "fuzzy_entropy"}:
        entropy_suffix = "_fuzzyen"
        entropy_tag = "FuzzyEn"
    else:
        raise ValueError(f"Unsupported --entropy: {args.entropy}")

    
    if args.window is not None:
        window_tag = str(args.window).rstrip('0').rstrip('.')
        if args.window == 0.5:
            args.data_path = rf'.\data_result\with_base_{window_tag}{entropy_suffix}'
            args.data_format = 'raw'
            args.segment = 0.5
        elif args.window == 1.0:
            args.data_path = rf'.\data_result\with_base_{window_tag}{entropy_suffix}'
            args.data_format = 'raw_1s'
            args.segment = 1.0
        elif args.window == 2.0:
            args.data_path = rf'.\data_result\with_base_{window_tag}{entropy_suffix}'
            args.data_format = 'raw_2s'
            args.segment = 2.0
        print('>>> window={}, entropy={}: data_path={}, data_format={}, segment={}'.format(
            args.window, entropy_tag, args.data_path, args.data_format, args.segment))

    sub_to_run = np.arange(args.subjects)

    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, feature=False, expand=False)

    if args.experiment == 'Dedx':
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        cv.dependent(subject=sub_to_run, fold=5, reproduce=args.reproduce)  
    elif args.experiment == 'Idex':
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        cv.independence(subject=sub_to_run, fold=10, reproduce=args.reproduce)
