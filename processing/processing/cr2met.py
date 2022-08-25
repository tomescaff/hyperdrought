import xarray as xr
import numpy as np
import pandas as pd
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
filename = 'CR2MET_pr_v2.0_mon_1979_2019_005deg.nc'
filepath = join(currentdir, '../../../hyperdrought_data/CR2MET', filename )

def get_cr2met_annual_precip():
    cr2met = xr.open_dataset(filepath, decode_times=False)['pr']
    cr2met['time'] = pd.date_range('1979-01-01', '2019-12-31', freq='1MS')
    cr2met_acc = cr2met.resample(time='1YS').sum(skipna=False)
    return cr2met_acc

def get_cr2met_MJJAS_precip():
    cr2met = xr.open_dataset(filepath, decode_times=False)['pr']
    cr2met['time'] = pd.date_range('1979-01-01', '2019-12-31', freq='1MS')
    mlist = [5,6,7,8,9]
    m = len(mlist)
    dlist = [31, 30, 31, 31, 30]
    dsum = sum(dlist)
    cr2met_mjjas = cr2met.where(cr2met.time.dt.month.isin(mlist), drop=True)
    cr2met_mjjas_acc = cr2met_mjjas.rolling(min_periods=m, time=m, center=True).sum('time')
    cr2met_mjjas_acc = cr2met_mjjas_acc.where(cr2met_mjjas_acc.time.dt.month==7, drop=True)
    cr2met_mjjas_mm_day = cr2met_mjjas_acc/dsum
    return cr2met_mjjas_mm_day

def get_cr2met_ONDJFMA_precip():
    cr2met = xr.open_dataset(filepath, decode_times=False)['pr']
    cr2met['time'] = pd.date_range('1979-01-01', '2019-12-31', freq='1MS')
    cr2met = cr2met.sel(time=slice('1979-10', '2019-4'))
    mlist = [10, 11, 12, 1, 2, 3, 4]
    m = len(mlist)
    dlist = [31, 30, 31, 31, 28, 31, 30]
    dsum = sum(dlist)
    cr2met_seas = cr2met.where(cr2met.time.dt.month.isin(mlist), drop=True)
    cr2met_seas_acc = cr2met_seas.rolling(min_periods=m, time=m, center=True).sum('time')
    cr2met_seas_acc = cr2met_seas_acc.where(cr2met_seas_acc.time.dt.month==1, drop=True)
    cr2met_seas_mm_day = cr2met_seas_acc/dsum
    return cr2met_seas_mm_day

def get_cr2met_JFM_precip_acc():
    cr2met = xr.open_dataset(filepath, decode_times=False)['pr']
    cr2met['time'] = pd.date_range('1979-01-01', '2019-12-31', freq='1MS')
    mlist = [1, 2, 3]
    m = len(mlist)
    cr2met_seas = cr2met.where(cr2met.time.dt.month.isin(mlist), drop=True)
    cr2met_seas_acc = cr2met_seas.rolling(min_periods=m, time=m, center=True).sum('time')
    cr2met_seas_acc = cr2met_seas_acc.where(cr2met_seas_acc.time.dt.month==2, drop=True)
    return cr2met_seas_acc


