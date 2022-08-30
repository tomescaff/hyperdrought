import pandas as pd
import xarray as xr 
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

# obtain Puerto Montt time series from CR2 explorer
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
