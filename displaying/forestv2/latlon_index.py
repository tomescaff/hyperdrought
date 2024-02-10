import pandas as pd
import numpy as np
import xarray as xr

# read data
filepath = "../../../hyperdrought_data/forestv2/ndvi_20012023_metadata.xlsx"
df = pd.read_excel(filepath) # read excel
n,m  = df.shape
meta = df.iloc[:,:4] # get metadata section

ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")

ilons = np.zeros((n,))
ilats = np.zeros((n,))

for i in range(n):
    
    alat = meta.iloc[i, 2]
    alon = meta.iloc[i, 3]
    ilons[i] = list(ds.lon.values).index(ds.sel(lon=alon, method='nearest').lon)
    ilats[i] = list(ds.lat.values).index(ds.sel(lat=alat, method='nearest').lat)
    
meta['ilon'] = ilons.astype(int)
meta['ilat'] = ilats.astype(int)

meta.to_csv('latlon_index.csv')