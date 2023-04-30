import numpy as np
def time2frame(t, fps):
    return int(np.floor(t * fps))