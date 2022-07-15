import xarray as xr
import numpy as np
import pandas as pd

def get_cr2met_annual_precip():

    cr2met = xr.open_dataset('../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.0_mon_1979_2019_005deg.nc', decode_times=False)
    cr2met['time'] = pd.date_range('1979-01-01', '2019-12-31', freq='1MS')
    cr2met_acc = cr2met.resample(time='1YS').sum(skipna=False)
    return cr2met_acc['pr']