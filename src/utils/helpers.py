# This code is modified from https://github.com/haoheliu/DCASE_2022_Task_5
import numpy as np
def time2frame(t, fps):
    if isinstance(t, list):
        return np.floor(np.array(t) * fps).astype(int)
    elif isinstance(t, float):
        return np.floor(t * fps).astype(int)
    else:
        print(t)
        raise TypeError('t should be float or list of float')
    
# GPT
def merge_intervals(meta):
    time_spane = meta['time_spane']
    if len(time_spane) == 0:
        return meta
    time_spane.sort(key=lambda x: x['start'])

    merged_intervals = []
    current_interval = time_spane[0]

    for i in range(1, len(time_spane)):
        if current_interval['end'] - time_spane[i]['start'] >= -0.05:
            current_interval['end'] = max(current_interval['end'], time_spane[i]['end'])
        else:
            merged_intervals.append(current_interval)
            current_interval = time_spane[i]
    merged_intervals.append(current_interval)
    meta['duration'] = []
    meta['time_spane'] = merged_intervals
    for i in meta['time_spane']:
        meta['duration'].append(i['end'] - i['start'])
    return meta
