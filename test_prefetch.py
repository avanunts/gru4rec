import argparse
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


from gru4rec import GRU4Rec
import prefetch
import data_util


parser = argparse.ArgumentParser(description='Build nn base for gru4rec model to use it during inference for comprising prefetch')
parser.add_argument('path', type=str, help='Path to the saved model')
parser.add_argument('-i', '--inference', metavar='INFERENCE_PATH', type=str, help='Path to the dataset to test prefetch located at INFERENCE_PATH.')
parser.add_argument('-nnbp', '--nn_base_path', metavar='NN_BASE_PATH', type=str, nargs='+', help='Two paths: to the nn base (i -> nn of i) and to the nn base index (list of item ids)')
parser.add_argument('-np', '--n_prefetch', metavar='N_PREFETCH', type=int, help='Number of items in prefetch.')
args = parser.parse_args()

print('Loading trained model from file: {}'.format(args.path))
gru = GRU4Rec.loadmodel(args.path)

tmp_dir = TemporaryDirectory()
f_path = args.inference
f_name = os.path.basename(f_path)
joined_input = pd.read_parquet(f_path)
joined_input['items'] = joined_input.apply(lambda x: np.append(x['items'], x['next_item']))
input_path = data_util.convert_joined_ds_and_store(joined_input, f_name, tmp_dir)
print('Loading inference data from path {}'.format(input_path))
input_data = data_util.load_data(input_path, gru)

print('Loading nn_base from path {}'.format(args.nn_base_path[0]))
nn_base = np.load(args.nn_base_path)
print('Loading nn_base idx from path {}'.format(args.nn_base_path[1]))
nn_base_idx = pd.read_parquet(args.nn_base_path[1]).id.values.tolist()

prefetch_ds = prefetch.build_prefetch(gru.itemidmap, input_data, nn_base, nn_base_idx, args.n_prefetch)
prefetch.test_prefetch(gru.itemidmap, input_data, prefetch_ds)

tmp_dir.cleanup()

