"""
@author: Balázs Hidasi
"""
import sys

import numpy as np
import pandas as pd
import theano
from theano import tensor as T


def infer_gpu(gru, test_data, prefetch_ds, session_key='SessionId', item_key='ItemId', time_key='Time', prefetch_key='Prefetch', cut_off=20, batch_size=100):
    '''
    Evaluates the GRU4Rec network quickly wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    prefetch_ds: pandas.DataFrame
        Prefetch. It contains prefetch item_idxs for each session. Then cut_off best items are selected from items(item_idxs).
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    prefetch_key : string
        Header of the prefetch item_idxs column in the prefetch_ds (default: 'Prefetch')
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    mode : 'standard', 'conservative', 'median', 'tiebreaking'
        Sets how ties (the exact same prediction scores) should be handled. Note that ties produced by GRU4Rec are very often a sign of saturation or some kind of error. 'standard' -> the positive item is ranked above all negatives with the same score; 'conservative' -> the positive item is ranked below all the negative items with the same score; 'median' -> assume that half of the negative items with the same score as the positive item are ranked before and the other half is ranked after, somewhat slower than the previous two; 'tiebreaking' -> add a small random value to every predicted score to break up ties, slowest of the modes. Default: 'standard'

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)

    '''
    validate_data(test_data, prefetch_ds)
    if gru.error_during_train: raise Exception
    print('Measuring Recall@{} and MRR@{}'.format(cut_off, cut_off))
    X = T.ivector()
    C = []
    H, updatesH = gru.symbolic_state(X, batch_size)

    eval_h = theano.function(inputs=[X] + C, updates=updatesH, allow_input_downcast=True)

    H_last = T.imatrix()
    prefetch = T.imatrix()
    yhat = gru.symbolic_predict_from_state(H_last, prefetch)
    prefetch_recoms = yhat.argsort()[:, ::-1][:, :cut_off]
    recoms = prefetch[np.arange(prefetch.shape[0]), prefetch_recoms]

    eval_recoms = theano.function(inputs=[H_last, prefetch] + C, outputs=[recoms])


    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx': gru.itemidmap.values, item_key: gru.itemidmap.index}),
                         on=item_key, how='inner')
    test_data = filter_sessions_of_length_1(test_data, session_key)
    test_data.sort_values([session_key, time_key, item_key], inplace=True)
    test_data_items = test_data.ItemIdx.values
    iters = np.arange(batch_size)
    maxiter = iters.max()
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    finished = False
    cidxs = []
    inference_sessions = np.zeros((test_data[session_key].nunique(), ), dtype=np.int32)
    inference_recoms = np.zeros((test_data[session_key].nunique(), cut_off), dtype=np.int32)
    inference_processed = 0
    while not finished:
        minlen = (end - start).min()
        out_idx = test_data_items[start]
        for i in range(minlen - 1):
            in_idx = out_idx
            out_idx = test_data_items[start + i + 1]
            eval_h(in_idx, *cidxs)
        start = start + minlen - 1
        finished_mask = (end - start <= 1)
        n_finished = finished_mask.sum()

        to_infer_sessions = test_data[session_key].values[start[finished_mask]]
        inference_sessions[inference_processed:inference_processed + n_finished] = to_infer_sessions
        prefetch_item_idxs = np.stack(prefetch_ds[prefetch_key].values[to_infer_sessions])
        H_last = H[-1].get_value(borrow=False)[finished_mask]
        recoms = eval_recoms(H_last, prefetch_item_idxs, *cidxs)
        inference_recoms[inference_processed:inference_processed + n_finished] = recoms
        inference_processed += n_finished
        iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
        maxiter += n_finished
        valid_mask = (iters < len(offset_sessions) - 1)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            finished = True
            break
        mask = finished_mask & valid_mask
        sessions = iters[mask]
        start[mask] = offset_sessions[sessions]
        end[mask] = offset_sessions[sessions + 1]
        iters = iters[valid_mask]
        start = start[valid_mask]
        end = end[valid_mask]
        if valid_mask.any():
            for i in range(len(H)):
                tmp = H[i].get_value(borrow=True)
                tmp[mask] = 0
                tmp = tmp[valid_mask]
                H[i].set_value(tmp, borrow=True)

    recom_ids = get_ids_from_indices(inference_recoms, gru.itemidmap).tolist()

    return pd.DataFrame({'SessionId': inference_sessions, 'prediction': recom_ids})


def get_ids_from_indices(indices, itemidmap):
    ids = itemidmap.index.values
    vectorized = np.vectorize(lambda x: ids[x])
    return vectorized(indices)


def filter_sessions_of_length_1(test_data, session_key):
    session_id, size = np.unique(test_data[session_key].values, return_counts=True)
    chosen_sessions = pd.DataFrame({session_key: session_id[size > 1]})
    return pd.merge(test_data, chosen_sessions, on=session_key, how='inner')


def validate_data(test_data, prefetch_ds, session_key, prefetch_key):
    test_data_session_ids = np.unique(test_data[session_key].values)
    max_session = test_data_session_ids.max()
    if not len(test_data_session_ids) == max_session + 1:
        print('test_data SessionIds must be of a form 0, ..., num_sessions - 1')
        sys.exit(1)
    if not (test_data_session_ids == np.arange(max_session + 1)).all():
        print('test_data SessionIds must be of a form 0, ..., num_sessions - 1')
        sys.exit(1)
    prefetch_session_ids = prefetch_ds[session_key].values
    if not len(prefetch_session_ids) == max_session + 1:
        print('prefetch SessionIds must be of a form 0, ..., num_sessions - 1')
        sys.exit(1)
    if not (prefetch_session_ids == np.arange(max_session + 1)).all():
        print('prefetch SessionIds must be of a form 0, ..., num_sessions - 1')
        sys.exit(1)
    prefetch_lens = np.unique(prefetch_ds[prefetch_key].apply(len).values)
    if not len(prefetch_lens) == 1:
        print('Prefetch must be of equal size for all sessions')
        sys.exit(1)
    if not prefetch_lens[0] >= 500:
        print('Prefetch len must be at least 500')
        sys.exit(1)
