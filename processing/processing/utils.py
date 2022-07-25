import xarray as xr
import pandas as pd
import numpy as np

# get return periods from arg as numpy array
def get_return_periods(z, method='up'):
    
    n = z.size

    # sort values
    z = np.sort(z)

    # get unique values
    u = np.unique(z)

    m = u.size

    # create matrix for tail probability and tau
    tail = np.zeros((m,))
    tau = np.zeros((m,))

    # compute tail and tau
    for i in range(m):
        x = u[i]
        if method == 'down':
            tail[i] = np.sum(z<=x)/n
        else: # up
            tail[i] = np.sum(z>=x)/n
        tau[i] = 1/tail[i]

    return u, tau