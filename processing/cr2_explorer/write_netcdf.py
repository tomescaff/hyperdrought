import pandas as pd
import numpy as np
import xarray as xr

stnfile = '../../../hyperdrought_data/CR2_explorer/cr2_prDaily_2022/cr2_prDaily_2022.txt'

df = pd.read_csv(stnfile, low_memory=False, index_col=0)

index = df.index.tolist()
n = len(index)
N = 14

cols = df.columns.tolist()
NCOLS = len(cols)

# coords
stn = np.arange(NCOLS)
date_range = pd.date_range('1900-01-01', '2022-12-31', freq='1D')
m = date_range.size

# data
lat = np.zeros((NCOLS,))
lon = np.zeros((NCOLS,))
alt = np.zeros((NCOLS,))
data = np.zeros((m, NCOLS))
name = []
inst = []

for i in range(NCOLS):
    col = df.iloc[:,i]

    lat[i] = float(col.loc['latitud'])
    lon[i] = float(col.loc['longitud'])
    alt[i] = float(col.loc['altura'])
    name = name + [col.loc['nombre']]
    inst = inst + [col.loc['institucion']]
    data[:,i] = col.iloc[N:]

da_lat = xr.DataArray(lat, coords=[stn], dims=['stn'])
da_lon = xr.DataArray(lon, coords=[stn], dims=['stn'])
da_alt = xr.DataArray(alt, coords=[stn], dims=['stn'])
da_name = xr.DataArray(name, coords=[stn], dims=['stn'])
da_inst = xr.DataArray(inst, coords=[stn], dims=['stn'])
da_data = xr.DataArray(data, coords=[date_range, stn], dims=['time','stn'])

ds = xr.Dataset({'lat':da_lat, 'lon':da_lon, 'alt':da_alt, 'name':da_name, 'institucion': da_inst, 'pr': da_data})
ds.to_netcdf('../../../hyperdrought_data/CR2_explorer/cr2_prDaily_2022.nc')