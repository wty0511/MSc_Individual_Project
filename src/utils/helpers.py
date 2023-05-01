import numpy as np
def time2frame(t, fps):
    if isinstance(t, list):
        return np.floor(np.array(t) * fps).astype(int)
    elif isinstance(t, float):
        return np.floor(t * fps).astype(int)
    else:
        print(t)
        raise TypeError('t should be float or list of float')
    