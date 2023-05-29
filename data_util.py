import os
import gc
import sys
import joblib

import pandas as pd
import numpy as np

JOINED = 'joined'  # format is {'SessionId': [0, 1], 'items': [['aa', 'bb'], ['cc']]} SessionId must be unique
EXPLODED = 'exploded'  # format is {'SessionId': [0, 0, 0, 1, 1], 'Time': [0, 1, 2, 0, 1], 'ItemId': ['aa', 'bb', 'cc', 'dd', 'ee']}


def load_data(fname, gru):
    if fname.endswith('.pickle'):
        print('Loading data from pickle file: {}'.format(fname))
        data = joblib.load(fname)
        if gru.session_key not in data.columns:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(gru.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if gru.item_key not in data.columns:
            print('ERROR. The column specified for item IDs "{}" is not in the data file ({})'.format(gru.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if gru.time_key not in data.columns:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(gru.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
    else:
        with open(fname, 'rt') as f:
            header = f.readline().strip().split('\t')
        if gru.session_key not in header:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(gru.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if gru.item_key not in header:
            print('ERROR. The colmn specified for item IDs "{}" is not in the data file ({})'.format(gru.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if gru.time_key not in header:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(gru.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
        print('Loading data from TAB separated file: {}'.format(fname))
        data = pd.read_csv(fname, sep='\t', usecols=[gru.session_key, gru.item_key, gru.time_key], dtype={gru.session_key:'int32', gru.item_key:np.str})
    return data


def convert_joined_ds_and_store(joined_ds_path, f_name, tmp_dir):
    joined = pd.read_parquet(joined_ds_path)
    result_path = os.path.join(tmp_dir.name, f_name)
    print('Temporal path for exploded ds: {}...'.format(result_path))
    print('Convert joined ds to exploded and save...')
    convert_joined_to_exploded(joined).to_csv(result_path, sep='\t')
    print('Free memory...')
    del joined
    gc.collect()
    return result_path


def convert_joined_to_exploded(joined):
    return explode_manually(joined)  # no explode in pandas=0.24.*


def explode_manually(ds):
    sessionIds = []
    times = []
    itemIds = []
    for i in range(ds.shape[0]):
        sessionId = ds.iloc[i]['SessionId']
        num_items = len(ds.iloc[i]['items'])
        sessionIds += [sessionId] * num_items
        times += list(range(num_items))
        itemIds += ds.iloc[i]['items'].tolist()
    return pd.DataFrame({'SessionId': sessionIds, 'Time': times, 'ItemId': itemIds})

