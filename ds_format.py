import pandas as pd

JOINED = 'joined'  # format is {'SessionId': [0, 1], 'items': [['aa', 'bb'], ['cc']]} SessionId must be unique
EXPLODED = 'exploded'  # format is {'SessionId': [0, 0, 0, 1, 1], 'Time': [0, 1, 2, 0, 1], 'ItemId': ['aa', 'bb', 'cc', 'dd', 'ee']}


def convert_joined_to_exploded(joined, with_inference, item_id_map):
    # didn't found any guarantees on preserving order during explode, so put time into exploding list
    time_and_session = joined.apply(lambda x: [[i, x] for i, x in enumerate(x.items)], axis=1)
    exploded = pd.DataFrame({
        'SessionId': joined.SessionId,
        'TimeItemId': time_and_session
    }).explode('TimeItemId')
    exploded['Time'] = exploded['TimeItemId'].apply(lambda x: x[0]).astype(int)
    exploded['ItemId'] = exploded['TimeItemId'].apply(lambda x: x[1]).astype(str)
    if not with_inference:
        return exploded[['SessionId', 'Time', 'ItemId']]
    return add_inference_column(exploded, item_id_map)


def add_inference_column(ds, item_id_map):
    filtered = pd.merge(ds, pd.DataFrame({'ItemId': item_id_map.index}), on='ItemId', how='inner')
    max_time = filtered[['SessionId', 'Time']].groupby('SessionId').max()
    max_time['to_infer'] = True
    result = filtered.merge(max_time, how='left', on=['SessionId', 'Time'])
    result.fillna(value=False, inplace=True)
    return result
