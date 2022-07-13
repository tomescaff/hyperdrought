import xarray as xr
import pandas as pd
import numpy as np

def from_mon_to_year(series_mon_ave, init_date='1850-01-01', end_date='2100-12-31'):
    
    dr = pd.date_range(init_date, end_date, freq='1D')
    one_per_day = xr.DataArray(np.ones((dr.size,)), coords=[dr], dims=['time'])
    days_per_month = one_per_day.resample(time='1MS').sum(skipna=False)
    days_per_year = one_per_day.resample(time='1YS').sum(skipna=False)

    series_mon_acc = series_mon_ave*days_per_month
    series_yea_acc = series_mon_acc.resample(time='1YS').sum(skipna=False)
    series_yea_ave = series_yea_acc/days_per_year

    return series_yea_ave, days_per_year

def get_LENS2_annual_precip(init_date='1850-01-01', end_date='2100-12-31'):

    ds = xr.open_dataset('../../../hyperdrought_data/LENS2_ALL/LENS2.PRECT.allruns.chile-fixedtime.nc')
    da = ds['PRECT']
    da_year, days_per_year = from_mon_to_year(da, init_date, end_date) # mean annual precip flux in m/s, days per year
    da_acc = da_year*3600*24*days_per_year*1000 # mean annual acc precip in mm
    return da_acc

def get_LENS2_annual_precip_QNEW():

    da = get_LENS2_annual_precip()
    qne = da.sel(lat= -33.455497, lon = (-70.0)%360, method='nearest').drop(['lat','lon'])
    qnw = da.sel(lat= -33.455497, lon = (-71.25)%360, method='nearest').drop(['lat','lon'])
    return (qnw + qne)/2.0
    

    