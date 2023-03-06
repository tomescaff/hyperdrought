import xarray as xr
import numpy as np
import pandas as pd
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
filename = 'CR2MET_pr_v2.5_mon_1960_2021_005deg.nc'
filepath = join(currentdir, '../../../hyperdrought_data/CR2MET', filename)

maskfile = 'CR2MET_clmask_v2.5_mon_1960_2021_005deg.nc'
maskpath = join(currentdir, '../../../hyperdrought_data/CR2MET', maskfile)

def get_cr2met_annual_precip():
    cr2met = xr.open_dataset(filepath, decode_times=False)['pr']
    cr2met['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
    cr2met_acc = cr2met.resample(time='1YS').sum(skipna=False)
    return cr2met_acc

def get_cl_mask():
    return xr.open_dataset(maskpath)['cl_mask']

