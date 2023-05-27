import argparse
import os
from tempfile import TemporaryDirectory

import numpy as np


from gru4rec import GRU4Rec
import prefetch
import data_util


parser = argparse.ArgumentParser(description='Build nn base for gru4rec model to use it during inference for comprising prefetch')
parser.add_argument('path', type=str, help='Path to the saved model')
parser.add_argument('-i', '--inference', metavar='INFERENCE_PATH', type=str, help='Path to the dataset to test prefetch located at INFERENCE_PATH.')
parser.add_argument('-nnbp', '--nn_base_path', type=str, help='Path to the file where to save nn base')
parser.add_argument('-np', '--n_prefetch', metavar='N_PREFETCH', type=int, help='Number of items in prefetch.')
args = parser.parse_args()

print('Loading trained model from file: {}'.format(args.path))
gru = GRU4Rec.loadmodel(args.path)

tmp_dir = TemporaryDirectory()
f_path = args.inference
f_name = os.path.basename(f_path)
input_path = data_util.convert_joined_ds_and_store(f_path, f_name, tmp_dir)
print('Loading inference data from path {}'.format(input_path))
input_data = data_util.load_data(input_path, gru)

print('Loading nn_base from path {}'.format(args.nn_base_path))
nn_base = np.load(args.nn_base_path)

prefetch_ds = prefetch.build_prefetch(gru.itemidmap, input_data, 'SessionId', 'ItemId', 'Time', 'Prefetch', nn_base, args.n_prefetch)
prefetch.test_prefetch(gru.itemidmap, input_data, prefetch_ds, 'SessionId', 'ItemId', 'Time', 'Prefetch')

tmp_dir.cleanup()

