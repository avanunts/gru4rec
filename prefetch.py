import sys
# from tqdm import tqdm
#
# import faiss
# import math

import pandas as pd
import numpy as np

# faiss_chunk_size = 20
n_last_item_for_nn = 5
# nn_map = {
#     2: [500, 500],
#     3: [500, 300, 200],
#     4: [500, 300, 100, 100],
#     5: [500, 200, 100, 100, 100],
# }

session_key = 'SessionId'
item_key = 'ItemId'
time_key = 'Time'
prefetch_key = 'Prefetch'



# def build_nn_base(item_id_map, vectors, vector_idx, nn_per_query):
#     merged_idx = pd.merge(
#         pd.DataFrame({'ItemIdx': item_id_map.values, item_key: item_id_map.index}),
#         pd.DataFrame({'VectorIdx': np.arange(len(vector_idx)), item_key: vector_idx}),
#         on=item_key, how='left', validate='1:1'
#     )
#     if merged_idx['VectorIdx'].isna().sum() > 0:
#         print('Vector base is not full')
#         sys.exit(1)
#     merged_idx.sort_values(['ItemIdx'], inplace=True)
#     vectors = vectors[merged_idx['VectorIdx'].values]
#     num_dim = vectors.shape[1]
#     index = faiss.IndexFlatL2(num_dim)
#     index.add(vectors)
#
#     num_items = vectors.shape[0]
#     num_chunks = math.ceil(num_items / faiss_chunk_size)
#     nn_base = np.zeros((num_items, nn_per_query), dtype=np.int32)
#     for i in tqdm(range(num_chunks)):
#         l, r = get_ith_chunk_indices(num_items, faiss_chunk_size, i)
#         _, nn_chunk = index.search(vectors[l: r], nn_per_query)
#         nn_base[l: r] = nn_chunk
#     return nn_base


def build_prefetch(item_id_map, input_data, nn_base, nn_base_idx, n_prefetch):
    aligned_nn_base = align_nn_base(item_id_map, nn_base, nn_base_idx)
    return build_prefetch_with_aligned_index(item_id_map, input_data, aligned_nn_base, n_prefetch)


def build_prefetch_with_aligned_index(item_id_map, input_data, nn_base, n_prefetch):
    df = pd.merge(input_data, pd.DataFrame({'ItemIdx': item_id_map.values, item_key: item_id_map.index}), on=item_key, how='inner')
    df.sort_values([session_key, time_key], inplace=True)
    validate_sessions(df)
    validate_time(df)
    prefetch_nn_keys_with_time = (
        filter_timeframes_for_nn(df)
        .groupby('SessionId')['TimeItemIdx'].apply(np.stack).values
    )
    num_sessions = prefetch_nn_keys_with_time.shape[0]
    prefetch_nn_values = np.zeros((num_sessions, n_prefetch), dtype=np.int32)
    for i in range(num_sessions):
        keys = extract_keys_from_time_item_idx(prefetch_nn_keys_with_time[i])
        extracted_prefetch = nn_base[keys].flatten() # expect not unique indices here
        prefetch_nn_values[i][:len(extracted_prefetch)] = extracted_prefetch # if extracted prefetch is too small, then left indices will be zero
    return pd.DataFrame({session_key: np.arange(num_sessions), prefetch_key: prefetch_nn_values.tolist()})


def test_prefetch(item_id_map, input_data, prefetch_ds):
    df = pd.merge(input_data, pd.DataFrame({'ItemIdx': item_id_map.values, item_key: item_id_map.index}), on=item_key, how='inner')
    session_id, session_size = np.unique(input_data[session_key].values, return_counts=True)
    max_time = pd.DataFrame({session_key: session_id, time_key: session_size - 1})
    df = pd.merge(df, max_time, on=[session_key, time_key], how='inner')[[session_key, 'ItemIdx']]
    recall = pd.merge(df, prefetch_ds, on=[session_key], how='inner').apply(lambda x: x['ItemIdx'] in x[prefetch_key], axis=1)
    print('TEST PREFETCH: total sessions {}, recall {}'.format(recall.shape[0], recall.values.mean()))


# def get_prefetch_values(nn_base, keys, n_prefetch):
#     if not len(keys) in nn_map.keys():
#         print('Too many keys for get prefetch: {}, but must be 2, 3 or 4'.format(len(keys)))
#         sys.exit(1)
#     supplied = set()
#     demand = nn_map[len(keys)]
#     left_demand = 0
#     result = np.zeros(n_prefetch, dtype=np.int32)
#     j = 0
#     for i in range(len(keys) - 1, -1, -1):
#         curr_demand = demand[i] + left_demand
#         new_supplied = [value for value in nn_base[keys[i]] if value not in supplied][:curr_demand]
#         result[j: j + len(new_supplied)] = new_supplied
#         j = j + len(new_supplied)
#         left_demand = curr_demand - len(new_supplied)
#         supplied.update(new_supplied)
#     return result


def get_ith_chunk_indices(num_queries, chunk_size, i):
    l = i * chunk_size
    r = min((i + 1) * chunk_size, num_queries)
    return l, r


def filter_timeframes_for_nn(df):
    session_id, session_size = np.unique(df[session_key].values, return_counts=True)
    left = df[session_key].apply(lambda x: max(session_size[x] - n_last_item_for_nn - 1, 0)).values
    right = df[session_key].apply(lambda x: session_size[x] - 1).values
    time = df[time_key].values
    res = df[(left <= time) & (time < right)]
    res['TimeItemIdx'] = res.apply(lambda x: [x[time_key], x['ItemIdx']], axis=1)
    return res[[session_key, 'TimeItemIdx']]


def validate_time(df):
    _, session_size = np.unique(df[session_key].values, return_counts=True)
    time = df[time_key].values
    if not (time == np.hstack([np.arange(s_size) for s_size in session_size])).all():
        print('Time must be 0, 1, ..., n - 1 for session of size n')
        sys.exit(1)


def validate_sessions(df):
    session_id, _ = np.unique(df[session_key].values, return_counts=True)
    if not (session_id == np.arange(len(session_id))).all():
        print('SessionId must be 0, 1, ..., num_sessions - 1')
        sys.exit(1)


def extract_keys_from_time_item_idx(x):
    return np.sort(x, axis=0)[::-1][:, 1]


def align_nn_base(item_id_map, nn_base, nn_base_idx):
    idx_map = pd.merge(
        pd.DataFrame({'gru_item_idx': item_id_map.values, 'item_id': item_id_map.index}),
        pd.DataFrame({'nn_item_idx': np.arange(len(nn_base_idx)), 'item_id': nn_base_idx}),
        on='item_id', how='outer'
    )
    gru_idx_to_nn_idx = idx_map.dropna().sort_values(by='gru_item_idx').nn_item_idx.values.astype(int)
    nn_idx_to_gru_idx = idx_map.fillna({'gru_item_idx': 0}).sort_values(by='nn_item_idx').gru_item_idx.values.astype(int)
    aligned_nn_base = np.zeros((len(gru_idx_to_nn_idx), nn_base.shape[1]))
    for gru_idx in range(len(gru_idx_to_nn_idx)):
        nn_idxs = nn_base[gru_idx_to_nn_idx[gru_idx]]
        aligned_nn_base[gru_idx] = nn_idx_to_gru_idx[nn_idxs]
    return aligned_nn_base
