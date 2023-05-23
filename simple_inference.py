"""
@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def infer_gpu(gru, test_data, items=None, session_key='SessionId', item_key='ItemId', time_key='Time', cut_off=20,
                 batch_size=100, mode='standard'):
    '''
    Evaluates the GRU4Rec network quickly wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
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
    if gru.error_during_train: raise Exception
    print('Measuring Recall@{} and MRR@{}'.format(cut_off, cut_off))
    srng = RandomStreams()
    X = T.ivector()
    Y = T.ivector()
    M = T.iscalar()
    C = []
    yhat, H, updatesH = gru.symbolic_predict(X, Y, M, items, batch_size)
    recoms = yhat.argsort()[:, ::-1][:, :cut_off]
    if mode == 'tiebreaking': yhat += srng.uniform(size=yhat.shape) * 1e-10
    if items is None:
        targets = T.diag(yhat.T[Y])
        others = yhat.T
    else:
        targets = T.diag(yhat.T[:M])
        others = yhat.T[M:]
    if mode == 'standard':
        ranks = (others > targets).sum(axis=0) + 1
    elif mode == 'conservative':
        ranks = (others >= targets).sum(axis=0)
    elif mode == 'median':
        ranks = (others > targets).sum(axis=0) + 0.5 * ((others == targets).sum(axis=0) - 1) + 1
    elif mode == 'tiebreaking':
        ranks = (others > targets).sum(axis=0) + 1
    else:
        raise NotImplementedError
    REC = (ranks <= cut_off).sum()
    MRR = ((ranks <= cut_off) / ranks).sum()
    evaluate = theano.function(inputs=[X, Y, M] + C, outputs=[REC, MRR, recoms, Y], updates=updatesH, allow_input_downcast=True,
                               on_unused_input='ignore')
    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx': gru.itemidmap.values, item_key: gru.itemidmap.index}),
                         on=item_key, how='inner')
    test_data.sort_values([session_key, time_key, item_key], inplace=True)
    test_data_items = test_data.ItemIdx.values
    if items is not None:
        item_idxs = gru.itemidmap[items]
    recall, mrr, n = 0, 0, 0
    iters = np.arange(batch_size)
    maxiter = iters.max()
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    finished = False
    cidxs = []
    inference_sessions = np.zeros((test_data[session_key].nunique(), ), dtype=np.int32)
    inference_recoms = np.zeros((test_data[session_key].nunique(), cut_off))
    inference_targets = np.zeros((test_data[session_key].nunique(), ), dtype=np.int32)
    inference_processed = 0
    while not finished:
        minlen = (end - start).min()
        out_idx = test_data_items[start]
        recoms = []
        targets = []
        for i in range(minlen - 1):
            in_idx = out_idx
            out_idx = test_data_items[start + i + 1]
            if items is not None:
                y = np.hstack([out_idx, item_idxs])
            else:
                y = out_idx
            rec, m, recoms, targets = evaluate(in_idx, y, len(iters), *cidxs)
            recall += rec
            mrr += m
            n += len(iters)
        start = start + minlen - 1
        finished_mask = (end - start <= 1)
        n_finished = finished_mask.sum()
        inference_sessions[inference_processed:inference_processed + n_finished] = test_data[session_key].values[start[finished_mask]]
        inference_recoms[inference_processed:inference_processed + n_finished] = [recoms[i] for i in np.where(finished_mask)[0]]
        inference_targets[inference_processed:inference_processed + n_finished] = [targets[i] for i in np.where(finished_mask)[0]]
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

    recom_ids = get_ids_from_indices(inference_recoms, gru.itemidmap)
    target_ids = get_ids_from_indices(inference_targets, gru.itemidmap)

    return recall / n, mrr / n, pd.DataFrame({'SessionId': inference_sessions, 'target': target_ids, 'prediction': recom_ids})


def get_ids_from_indices(indices, itemidmap):
    ids = itemidmap.index.values
    vectorized = np.vectorize(lambda x: ids[x])
    return vectorized(indices)