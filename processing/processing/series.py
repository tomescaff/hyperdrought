import pandas as pd
import xarray as xr 

# obtain QN, RPIv1 and RPIv2 time series from RG data
def get_QN_RPI():
    filepath = '../../../hyperdrought_data/series/tseries_QN_RPIs_3037_1850_2021.txt'
    df = pd.read_csv(filepath, sep='\s+', header = None)
    df = df.rename({0:'QN', 1:'RPIv1', 2:'RPIv2'}, axis='columns')
    dr = pd.date_range('1850', '2021', freq='1YS')
    df['time'] = dr
    df = df.set_index('time')
    ds = df.to_xarray()
    return ds

# obtain QN time series from CR2 explorer
def get_QN_annual_precip():
    filepath = '../../../hyperdrought_data/series/QN_annual_precip_CR2.csv'
    df = pd.read_csv(filepath, delimiter=",", decimal=".", parse_dates={'time': ['agno', ' mes', ' dia']})
    df = df.rename({' valor':'precip'}, axis='columns')
    df = df.set_index('time')
    da = df['precip'].to_xarray()
    return da

def get_QN_annual_precip_long_record():
    filepath = '../../../hyperdrought_data/series/SANTIAGO_QN_1866_2020_RENE_ext.csv'
    df = pd.read_csv(filepath, delimiter=";", decimal=".", parse_dates=['FECHA'])
    months = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
    df_sum = df[months].sum(axis=1)
    da = xr.DataArray(df_sum, coords=[df['FECHA']], dims=['time'])
    return da 
