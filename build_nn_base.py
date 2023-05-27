import argparse

import numpy as np

from gru4rec import GRU4Rec
import prefetch


parser = argparse.ArgumentParser(description='Build nn base for gru4rec model to use it during inference for comprising prefetch')
parser.add_argument('path', type=str, help='Path to the saved model')
parser.add_argument('-nnbp', '--nn_base_path', type=str, help='Path to the file where to save nn base')
parser.add_argument('-vp', '--vectors_path', type=str, help='Path to the vectors numpy 2d array')
parser.add_argument('-vip', '--vector_index_path', type=str, help='Path to the vector idx')
parser.add_argument('-nnpq', '--nn_per_query', type=int, help='How many nearest neighbours to save in the base')
args = parser.parse_args()

print('Loading trained model from file: {}'.format(args.path))
gru = GRU4Rec.loadmodel(args.path)
print('Loading vectors from file: {}'.format(args.vectors_path))
vectors = np.load(args.vectors_path).astype(np.float32)
print('Loading vector index from file: {}'.format(args.vector_index_path))
vector_idx = np.load(args.vector_index_path)
nn_per_query = args.nn_per_query
print('Start building nn base...')
nn_base = prefetch.build_nn_base(gru.itemidmap, vectors, vector_idx, nn_per_query, item_key='ItemId')
np.save(args.nn_base_path, nn_base)
print('Saved nn base to {}'.format(args.nn_base_path))






