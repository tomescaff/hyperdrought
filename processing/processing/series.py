import pandas as pd
import xarray as xr 

def get_QN_RPI():
    filepath = '../../../hyperdrought_data/series/tseries_QN_RPIs_3037_1850_2021.txt'
    df = pd.read_csv(filepath, sep='\s+', header = None)
    df = df.rename({0:'QN', 1:'RPIv1', 2:'RPIv2'}, axis='columns')
    dr = pd.date_range('1850', '2021', freq='1YS')
    df['time'] = dr
    df = df.set_index('time')
    ds = df.to_xarray()
    return ds