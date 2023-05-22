import sys
import numpy as np
import pandas as pd
from collections import OrderedDict
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SessionsTraverser:
    def __init__(self, data, batch_size, session_key='SessionId'):
        if not np.all(np.diff(data[session_key].values) >= 0):
            print('Sessions traverser error: {} column must be sorted'.format(session_key))
            sys.exit(1)
        num_sessions = data[session_key].nunique()
        self.starts = np.zeros(num_sessions, dtype=np.int32)
        _, session_sizes = np.unique(data['SessionId'], return_counts=True)
        self.ends = session_sizes.cumsum() - 1  # inclusive ends
        self.starts[1:] = self.ends[:-1] + 1
        self.next_batch = np.zeros(batch_size, dtype=np.int32) - 1
        self.next_batch[:min(batch_size, num_sessions)] = self.starts[:batch_size]
        self.next_batch_termination = np.zeros(batch_size, dtype=np.int32) - 1
        self.next_batch_termination[:min(batch_size, num_sessions)] = self.ends[:batch_size]
        self.last_session = min(batch_size - 1, num_sessions - 1)
        self.num_sessions = num_sessions

    def get_next(self):
        next_indices = self.next_batch.copy()
        is_at_finish = np.array([False] * next_indices.shape[0])
        for i in range(self.next_batch.shape[0]):
            if self.next_batch[i] == -1:
                continue
            if self.next_batch[i] == self.next_batch_termination[i]:
                is_at_finish[i] = True
                if self.last_session == self.num_sessions - 1:
                    self.next_batch[i] = -1
                    self.next_batch_termination[i] = -1
                else:
                    self.last_session += 1
                    self.next_batch[i] = self.starts[self.last_session]
                    self.next_batch_termination[i] = self.ends[self.last_session]
            else:
                self.next_batch[i] += 1
        return next_indices, is_at_finish

    def empty(self):
        return np.all(self.next_batch == -1)


def infer_gpu(gru, input_data, session_key='SessionId', item_key='ItemId', time_key='Time', cut_off=20,
              batch_size=100, mode='standard'):
    '''
    Infers the GRU4Rec network quickly

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    input_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
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
    srng = RandomStreams()
    X = T.ivector()
    Y = T.ivector()
    M = T.iscalar()
    yhat, H, updatesH = gru.symbolic_predict(X, Y, M, None, batch_size)
    if mode == 'tiebreaking': yhat += srng.uniform(size=yhat.shape) * 1e-10

    infer = theano.function(inputs=[X, Y, M], outputs=[yhat], updates=updatesH, allow_input_downcast=True,
                               on_unused_input='ignore')

    input_data = pd.merge(input_data, pd.DataFrame({'ItemIdx': gru.itemidmap.values, item_key: gru.itemidmap.index}),
                          on=item_key, how='inner')
    input_data.sort_values([session_key, time_key, item_key], inplace=True)
    traverser = SessionsTraverser(input_data, batch_size)
    blank_idx = 0
    num_sessions = input_data[session_key].nunique()
    session_idx = pd.Series(data=np.arange(num_sessions), index=input_data[session_key].unique())
    scores = np.zeros((num_sessions, gru.itemidmap.values.max() + 1))
    while not traverser.empty():
        next_indices, is_at_finish = traverser.get_next()
        blank_indices = np.where(next_indices < 0)[0]
        next_indices[blank_indices] = blank_idx
        in_idx = input_data.ItemIdx.values[next_indices]
        curr_scores = infer(in_idx, None, None)[0]
        if is_at_finish.any():
            finished_session_ids = input_data.loc[next_indices[is_at_finish], 'SessionId'].values
            finished_session_idxs = session_idx[finished_session_ids].values
            finished_session_scores = [curr_scores[i] for i in np.where(is_at_finish)[0]]
            scores[finished_session_idxs] = finished_session_scores
            for i in range(len(H)):
                tmp = H[i].get_value(borrow=True)
                tmp[np.where(is_at_finish)[0]] = 0
    indices = get_indices_from_scores(scores, cut_off)
    ids = get_ids_from_indices(indices, gru.itemidmap).tolist()
    return pd.DataFrame({'SessionId': session_idx.index, 'prediction': ids})


def get_indices_from_scores(scores, recom_size):
    return np.argsort(scores, axis=1)[:, ::-1][:, :recom_size]


def get_ids_from_indices(indices, itemidmap):
    ids = itemidmap.index.values
    vectorized = np.vectorize(lambda x: ids[x])
    return vectorized(indices)
