import pandas as pd
import xarray as xr 
import numpy as np
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/series/'

# obtain QN, RPIv1 and RPIv2 time series from RG data
def get_QN_RPI():
    filename = 'tseries_QN_RPIs_3037_1850_2021.txt'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, sep='\s+', header = None)
    df = df.rename({0:'QN', 1:'RPIv1', 2:'RPIv2'}, axis='columns')
    dr = pd.date_range('1850', '2021', freq='1YS')
    df['time'] = dr
    df = df.set_index('time')
    ds = df.to_xarray()
    return ds

# obtain QN time series from CR2 explorer
def get_QN_annual_precip():
    filename = 'QN_annual_precip_CR2.csv'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=",", decimal=".", parse_dates={'time': ['agno', ' mes', ' dia']})
    df = df.rename({' valor':'precip'}, axis='columns')
    df = df.set_index('time')
    da = df['precip'].to_xarray()
    return da

def get_QN_annual_precip_long_record():
    filename = 'SANTIAGO_QN_1866_2020_RENE_ext.csv'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=";", decimal=".", parse_dates=['FECHA'])
    months = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
    df_sum = df[months].sum(axis=1)
    da = xr.DataArray(df_sum, coords=[df['FECHA']], dims=['time'])
    return da 

# obtain CU time series from CR2 explorer
def get_CU_annual_precip():
    filename = 'DMC_acc_annual_precip_curico_90p.csv'
    relpath = '../../../hyperdrought_data/DMC/'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=",", decimal=".", parse_dates={'time': ['agno', ' mes', ' dia']})
    df = df.rename({' valor':'precip'}, axis='columns')
    df = df.set_index('time')
    da = df['precip'].to_xarray()
    return da

# obtain Puerto Montt JFM time series from CR2 explorer
def get_PM_JFM_precip():
    filename = 'PM_JFM_precip_CR2.csv'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=",", decimal=".", parse_dates={'time': ['agno', ' mes', ' dia']})
    df = df.rename({' valor':'precip'}, axis='columns')
    df = df.set_index('time')
    da = df['precip'].to_xarray()
    da = da.sel(time=slice('1950','2021'))
    da = da.resample(time='1YS').sum('time')
    return da

# obtain Puerto Montt monthly time series from CR2 explorer
def get_PM_mon_precip():
    filename = 'PM_mon_precip_CR2.csv'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=",", decimal=".", parse_dates={'time': ['agno', ' mes', ' dia']})
    df = df.rename({' valor':'precip'}, axis='columns')
    df = df.set_index('time')
    da = df['precip'].to_xarray()
    da = da.sel(time=slice('1950','2021'))
    return da

# obtain Coyhaique time series from CR2 explorer
def get_CO_JFM_precip():
    filename = 'CO_JFM_precip_CR2.csv'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=",", decimal=".", parse_dates={'time': ['agno', ' mes', ' dia']})
    df = df.rename({' valor':'precip'}, axis='columns')
    df = df.set_index('time')
    da = df['precip'].to_xarray()
    da = da.sel(time=slice('1954','2021'))
    da = da.resample(time='1YS').sum('time')
    return da

# obtain MJJAS Nino3.4 anomaly time series from CPC
def get_MJJAS_NINO34a():
    filename = 'ersst5.nino.mth.91-20.ascii.txt'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, sep='\s+', parse_dates={'time': ['YR', 'MON']})
    df = df.rename({'ANOM.3':'NINO3.4a'}, axis='columns')
    df = df.set_index('time')
    da = df['NINO3.4a'].to_xarray()
    da = da.sel(time=slice('1950', '2021'))
    da = da.where(da.time.dt.month.isin([5,6,7,8,9]), drop=True)
    da = da.resample(time='1YS').mean('time')
    return da

def get_MJJAS_Nino34_long_record():
    filename = 'nino34.long.anom.data.txt'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, sep='\s+', skiprows=1, skipfooter=7, engine='python', header=None)
    iniyear = str(int(df.iloc[0,0]))
    endyear = str(int(df.iloc[-1,0]))
    datetime = pd.date_range(start=iniyear+'-01-01', end=endyear+'-12-31', freq='MS')
    da = xr.DataArray(np.ravel(df.iloc[:,1:]), coords=[datetime], dims=['time'])
    da[da==-99.99] = np.nan
    da = da.sel(time=slice('1870', '2021'))
    da = da.where(da.time.dt.month.isin([5,6,7,8,9]), drop=True)
    da = da.resample(time='1YS').mean('time')
    return da

def get_QN_MJJAS_precip_long_record():
    filename = 'SANTIAGO_QN_1866_2020_RENE_ext.csv'
    filepath = join(currentdir, relpath, filename)
    df = pd.read_csv(filepath, delimiter=";", decimal=".", parse_dates=['FECHA'])
    months = ['MAY', 'JUN', 'JUL', 'AGO', 'SEP']
    df_sum = df[months].sum(axis=1)
    da = xr.DataArray(df_sum, coords=[df['FECHA']], dims=['time'])
    return da 

