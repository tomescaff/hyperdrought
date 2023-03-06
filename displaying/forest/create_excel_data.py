import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

df = pd.read_excel("../../../hyperdrought_data/forest/nbr_shp_19112022.dbf.xlsx")
df_n, df_m = df.shape

columns = list(df.columns)
columns_data = columns[4:]
dfout = pd.DataFrame(columns=df.columns, index=df.index)

###############
# ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
# data = ds['pr']
ds = xr.open_dataset('../../../hyperdrought_data/CR2MET/CR2MET_tmin_tmax_v2.5_mon_1960_2021_005deg.nc', decode_times=False)
ds['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
data = ds['tmax']
###############

lat = data.lat
lon = data.lon

summer_months = data.sel(time=slice('1960-10-01', '2021-09-30')).where(data.time.dt.month.isin([10,11,12,1,2,3,4]), drop=True)
winter_months = data.sel(time=slice('1960-07-01', '2021-10-31')).where(data.time.dt.month.isin([7,8,9,10]), drop=True)

###############
# summer_series = summer_months.rolling(time=7, center=True).sum()[3::7]
# winter_series = winter_months.rolling(time=4, center=True).sum()[2::4]
summer_series = summer_months.rolling(time=7, center=True).mean()[3::7]
winter_series = winter_months.rolling(time=4, center=True).mean()[2::4]
##############

for i in range(df_n):

    print(i)
    
    # ilon = list(ds.lon.values).index(ds.sel(lon=row['long'], method='nearest').lon)
    # ilat = list(ds.lat.values).index(ds.sel(lat=row['lat'], method='nearest').lat)

    row = df.iloc[i,:]
    dfout.iloc[i,0] =row['ID']
    dfout.iloc[i,1] =row['count']
    dfout.iloc[i,2] =row['lat']
    dfout.iloc[i,3] =row['long']

    for j in range(len(columns_data)):

        label = columns_data[j]
        year = label[1:5]
        seas = label[-2:]

        if year == '2022':
            continue

        if seas == '01':
            dfout.iloc[i,j+4] = float(summer_series.sel(time=year, lat=row['lat'], lon=row['long'], method='nearest').values)
        else:
            dfout.iloc[i,j+4] = float(winter_series.sel(time=year, lat=row['lat'], lon=row['long'], method='nearest').values)

dfout.to_excel("../../../hyperdrought_data/forest/tmax_data.xlsx", sheet_name='nbr_shp1', index=False)


