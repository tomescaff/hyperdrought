import xarray as xr
import numpy as np
import pandas as pd
from os.path import join, abspath, dirname
from shapely import geometry

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

def get_regional_mask():
    
    cr2met_mask = get_cl_mask()
    mask = cr2met_mask.where((cr2met_mask.lat >= -36) & (cr2met_mask.lat <= -32))
    return mask

def get_regional_shape():

    mask = get_regional_mask()
    lat = mask.lat
    lon = mask.lon
    n, m = mask.shape
    left = []
    right = []
    for i in range(n):
        valid = False
        for j in range(m):
            point = not np.isnan(mask[i,j].values)
            if point is True and valid is False:
                left.append((float(lon[j].values), float(lat[i].values)))
                valid = True
            if point is True and valid is True:
                continue
            if point is False and valid is True:
                right.append((float(lon[j-1].values), float(lat[i].values)))
                valid = False
            if point is False and valid is False:
                continue
    points = left + right[::-1] + [left[0]]
    poly = geometry.Polygon(points)
    return poly

def get_regional_index():

    mask = get_regional_mask()
    cr2met = get_cr2met_annual_precip()
    cr2met_masked = cr2met*mask
    series = cr2met_masked.mean(['lat', 'lon'])
    return series

def get_cr2met_JFM_precip_acc():
    cr2met = xr.open_dataset(filepath, decode_times=False)['pr']
    cr2met['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
    mlist = [1, 2, 3]
    m = len(mlist)
    cr2met_seas = cr2met.where(cr2met.time.dt.month.isin(mlist), drop=True)
    cr2met_seas_acc = cr2met_seas.rolling(min_periods=m, time=m, center=True).sum('time')
    cr2met_seas_acc = cr2met_seas_acc.where(cr2met_seas_acc.time.dt.month==2, drop=True)
    return cr2met_seas_acc