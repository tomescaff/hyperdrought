import pandas as pd
import xarray as xr 

filepath = '../../../hyperdrought_data/series/SANTIAGO_QN_1866_2020_RENE_ext.csv'
df = pd.read_csv(filepath, delimiter=";", decimal=".", parse_dates=['FECHA'])
months = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
df_sum = df[months].sum(axis=1)
da = xr.DataArray(df_sum, coords=[df['FECHA']], dims=['time'])
