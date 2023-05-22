import pandas as pd

JOINED = 'joined'  # format is {'SessionId': [0, 1], 'items': [['aa', 'bb'], ['cc']]} SessionId must be unique
EXPLODED = 'exploded'  # format is {'SessionId': [0, 0, 0, 1, 1], 'Time': [0, 1, 2, 0, 1], 'ItemId': ['aa', 'bb', 'cc', 'dd', 'ee']}


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

