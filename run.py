import argparse
import shutil
from tempfile import TemporaryDirectory

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train or load a GRU4Rec model & measure recall and MRR on the specified test set(s).')
parser.add_argument('path', metavar='PATH', type=str, help='Path to the training data (TAB separated file (.tsv or .txt) or pickled pandas.DataFrame object (.pickle)) (if the --load_model parameter is NOT provided) or to the serialized model (if the --load_model parameter is provided).')
parser.add_argument('-f', '--format', type=str, default='exploded', help='Which format to use, options are: joined or exploded. In exploded a tab-separated dataset is expected. In joined a parquet file.')
parser.add_argument('-ps', '--parameter_string', metavar='PARAM_STRING', type=str, help='Training parameters provided as a single parameter string. The format of the string is `param_name1=param_value1,param_name2=param_value2...`, e.g.: `loss=bpr-max,layers=100,constrained_embedding=True`. Boolean training parameters should be either True or False; parameters that can take a list should use / as the separator (e.g. layers=200/200). Mutually exclusive with the -pf (--parameter_file) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-pf', '--parameter_file', metavar='PARAM_PATH', type=str, help='Alternatively, training parameters can be set using a config file specified in this argument. The config file must contain a single OrderedDict named `gru4rec_params`. The parameters must have the appropriate type (e.g. layers = [100]). Mutually exclusive with the -ps (--parameter_string) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-l', '--load_model', action='store_true', help='Load an already trained model instead of training a model. Mutually exclusive with the -ps (--parameter_string) and the -pf (--parameter_file) arguments and one of the three must be provided.')
parser.add_argument('-s', '--save_model', metavar='MODEL_PATH', type=str, help='Save the trained model to the MODEL_PATH. (Default: don\'t save model)')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, nargs='+', help='Path to the test data set(s) located at TEST_PATH. Multiple test sets can be provided (separate with spaces). (Default: don\'t evaluate the model)')
parser.add_argument('-i', '--inference', metavar='INFERENCE_PATH', type=str, nargs='+', help='Paths to the datasets to infer model located at INFERENCE_PATH. Must be in format \'p1 p2 ... p2k-1 p2k\', where p2m-1 is input and p2m is output')
parser.add_argument('-nnbp', '--nn_base_path', metavar='NN_BASE_PATH', type=str, help='Path to the nn base')
parser.add_argument('-np', '--n_prefetch', metavar='N_PREFETCH', type=int, help='Number of items in prefetch.')
parser.add_argument('-ibs', '--inference_batch_size', metavar='INFERENCE_BATCH_SIZE', type=int, help='Batch size during inference.')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='+', default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median', 'tiebreaking'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of evaluate_gpu() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-ss', '--sample_store_size', metavar='SS', type=int, default=10000000, help='GRU4Rec uses a buffer for negative samples during training to maximize GPU utilization. This parameter sets the buffer length. Lower values require more frequent recomputation, higher values use more (GPU) memory. Unless you know what you are doing, you shouldn\'t mess with this parameter. (Default: 10000000)')
parser.add_argument('--sample_store_on_cpu', action='store_true', help='If provided, the sample store will be stored in the RAM instead of the GPU memory. This is not advised in most cases, because it significantly lowers the GPU utilization. This option is provided if for some reason you want to train the model on the CPU (NOT advised).')
parser.add_argument('--test_against_items', metavar='N_TEST_ITEMS', type=int, help='It is NOT advised to evaluate recommender algorithms by ranking a single positive item against a set of sampled negatives. It overestimates recommendation performance and also skewes comparisons, as it affects algorithms differently (and if a different sequence of random samples is used, the results are downright uncomparable). If testing takes too much time, it is advised to sample test sessions to create a smaller test set. However, if the number of items is very high (i.e. ABOVE FEW MILLIONS), it might be impossible to evaluate the model within a reasonable time, even on a smaller (but still representative) test set. In this case, and this case only, one can sample items to evaluate against. This option allows to rank the positive item against the N_TEST_ITEMS most popular items. This has a lesser effect on comparison and it is a much stronger criteria than ranking against randomly sampled items. Keep in mind that the real performcance of the algorithm will still be overestimated by the results, but comparison will be mostly fair. If used, you should NEVER SET THIS PARAMETER BELOW 50000 and try to set it as high as possible (for your required evaluation time). (Default: all items are used as negatives for evaluation)')
args = parser.parse_args()

import os.path
orig_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sys
import time
from collections import OrderedDict
from gru4rec import GRU4Rec
import data_util
import evaluation
import inference
import prefetch
import importlib.util
os.chdir(orig_cwd)


if (args.parameter_string is not None) + (args.parameter_file is not None) + (args.load_model) != 1:
    print('ERROR. Exactly one of the following parameters must be provided: --parameter_string, --parameter_file, --load_model')
    sys.exit(1)

if args.format == data_util.JOINED:
    tmp_dir = TemporaryDirectory()

if args.load_model:
    print('Loading trained model from file: {}'.format(args.path))
    gru = GRU4Rec.loadmodel(args.path)
else:
    if args.parameter_file:
        param_file_path = os.path.abspath(args.parameter_file)
        param_dir, param_file = os.path.split(param_file_path)
        spec = importlib.util.spec_from_file_location(param_file.split('.py')[0], os.path.abspath(args.parameter_file))
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        gru4rec_params = params.gru4rec_params
        print('Loaded parameters from file: {}'.format(param_file_path))
    if args.parameter_string:
        gru4rec_params = OrderedDict([x.split('=') for x in args.parameter_string.split(',')])
    gru = GRU4Rec()
    gru.set_params(**gru4rec_params)
    if args.format == data_util.JOINED:
        train_path = data_util.convert_joined_ds_and_store(args.path, 'train.tsv', tmp_dir)
    elif args.format == data_util.EXPLODED:
        train_path = args.path
    else:
        print('--format (-f) option must be one of two: {}/{}, but it is {}.'.format(data_util.JOINED, data_util.EXPLODED, args.format))
        sys.exit(1)
    print('Loading training data...')
    data = data_util.load_data(train_path, gru)
    store_type = 'cpu' if args.sample_store_on_cpu else 'gpu'
    if store_type == 'cpu':
        print('WARNING! The sample store is set to be on the CPU. This will make training significantly slower on the GPU.')
    print('Started training')
    t0 = time.time()
    gru.fit(data, sample_store=args.sample_store_size, store_type='gpu')
    t1 = time.time()
    print('Total training time: {:.2f}s'.format(t1 - t0))
    if args.save_model is not None:
        print('Saving trained model to: {}'.format(args.save_model))
        gru.savemodel(args.save_model)

items = None
if args.test_against_items is not None:
    if args.test_against_items < 50000:
        print('ERROR. You musn\'t evaluate positive items agains less than 50000 items.')
        sys.exit(1)
    print('WARNING! You set the number of negative test items. You musn\'t evaluate positive items against a subset of all items unless the number of items in your data is too high (i.e. above a few millions) and evaluation takes too much time.')
    supp = data.groupby('ItemId').size()
    supp.sort_values(inplace=True, ascending=False)
    items = supp[:args.test_against_items].index

if (args.test is not None) + (args.inference is not None) > 1:
    print('ERROR. Maximum one of the following parameters must be provided: --test, --inference')
    sys.exit(1)

if args.test is not None:
    for test_file in args.test:
        if args.format == data_util.JOINED:
            test_path = data_util.convert_joined_ds_and_store(test_file, 'test.tsv', tmp_dir)
        else:
            test_path = test_file
        print('Loading test data...')
        test_data = data_util.load_data(test_path, gru)
        for c in args.measure:
            print('Starting evaluation (cut-off={}, using {} mode for tiebreaking)'.format(c, args.eval_type))
            t0 = time.time()
            res = evaluation.evaluate_gpu(gru, test_data, items, batch_size=100, cut_off=c, mode=args.eval_type)
            t1 = time.time()
            print('Evaluation took {:.2f}s'.format(t1 - t0))
            print('Recall@{}: {:.6f} MRR@{}: {:.6f}'.format(c, res[0], c, res[1]))

if args.inference is not None:
    if len(args.measure) != 1:
        print('Must use only one cutoff for inference, got list {}'.format(args.measure))
        sys.exit(1)
    if len(args.inference) % 2 != 0:
        print('--inference (-i) must contain even number of paths: 2i-1 for input and 2i for output')
        sys.exit(1)
    if args.format != data_util.JOINED:
        print('Must use inference only with --format (-f) set to joined, but use with {} instead'.format(args.format))
        sys.exit(1)
    if args.test_against_items is not None:
        print('Option --test_against_items during inference is not supported during inference, but it is assigned {}'.format(args.test_against_items))
        sys.exit(1)
    if args.nn_base_path is None or args.n_prefetch is None:
        print('Options --nn_base_path (-nnbp) and --n_prefetch (-np) must be assigned during inference, but at least one of them is None')
        sys.exit(1)
    if not os.path.exists(args.nn_base_path):
        print('''NN base doesn't exist for the model, please build nn base with build_nn_base.py first''')
        sys.exit(1)
    for i in range(int(len(args.inference) / 2)):
        f_path = args.inference[2 * i]
        f_name = os.path.basename(f_path)
        input_path = data_util.convert_joined_ds_and_store(f_path, f_name, tmp_dir)
        output_path = args.inference[2 * i + 1]
        print('Loading inference data from path {}'.format(input_path))
        input_data = data_util.load_data(input_path, gru)
        c = args.measure[0]
        print('Start building prefetch...')
        t0 = time.time()
        nn_base = np.load(args.nn_base_path)
        prefetch_ds = prefetch.build_prefetch(gru.itemidmap, input_data, 'SessionId', 'ItemId', 'Time', 'Prefetch', nn_base, args.n_prefetch)
        t1 = time.time()
        print('End building prefetch, took {:.2f}s'.format(t1 - t0))
        print('Starting inference (cut-off={}, using {} mode for tiebreaking)'.format(c, args.eval_type))
        t0 = time.time()
        results = inference.infer_gpu(gru, input_data, prefetch_ds, batch_size=args.inference_batch_size, cut_off=c)
        t1 = time.time()
        print('Inference took {:.2f}s'.format(t1 - t0))
        print('Saving results to {}'.format(output_path))
        t2 = time.time()
        results.to_parquet(output_path, engine='pyarrow')
        t3 = time.time()
        print('Saving took {:.2f}s'.format(t3 - t2))

if args.format == data_util.JOINED:
    tmp_dir.cleanup()
