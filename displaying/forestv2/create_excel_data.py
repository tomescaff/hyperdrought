import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

df = pd.read_excel("../../../hyperdrought_data/forestv2/ndvi_20012023_metadata.xlsx")
df_n, df_m = df.shape

columns = list(df.columns)
columns_data = columns[4:]

###############
# tmax data 
dfout = pd.DataFrame(columns=df.columns, index=df.index)
ds = xr.open_dataset('../../../hyperdrought_data/CR2MET/CR2MET_tmin_tmax_v2.5_mon_1960_2021_005deg.nc', decode_times=False)
ds['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
data = ds['tmax']
lat = data.lat
lon = data.lon

summer_months = data.sel(time=slice('1985-11-01', '2021-03-31')).where(data.time.dt.month.isin([11,12,1,2,3]), drop=True)
winter_months = data.sel(time=slice('1986-05-01', '2021-09-30')).where(data.time.dt.month.isin([5,6,7,8,9]), drop=True)

summer_series = summer_months.rolling(time=5, center=True).mean()[2::5]
winter_series = winter_months.rolling(time=5, center=True).mean()[2::5]

for i in range(df_n):
    
    print(i, df_n)
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
dfout.to_excel("../../../hyperdrought_data/forestv2/tmax_data.xlsx", index=False)
###############

###############
# tmin data 
dfout = pd.DataFrame(columns=df.columns, index=df.index)
ds = xr.open_dataset('../../../hyperdrought_data/CR2MET/CR2MET_tmin_tmax_v2.5_mon_1960_2021_005deg.nc', decode_times=False)
ds['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
data = ds['tmin']
lat = data.lat
lon = data.lon

summer_months = data.sel(time=slice('1985-11-01', '2021-03-31')).where(data.time.dt.month.isin([11,12,1,2,3]), drop=True)
winter_months = data.sel(time=slice('1986-05-01', '2021-09-30')).where(data.time.dt.month.isin([5,6,7,8,9]), drop=True)

summer_series = summer_months.rolling(time=5, center=True).mean()[2::5]
winter_series = winter_months.rolling(time=5, center=True).mean()[2::5]

for i in range(df_n):
    
    print(i, df_n)
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
dfout.to_excel("../../../hyperdrought_data/forestv2/tmin_data.xlsx", index=False)
###############

###############
# pr data 
dfout = pd.DataFrame(columns=df.columns, index=df.index)
ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
ds['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
data = ds['pr']
lat = data.lat
lon = data.lon

summer_months = data.sel(time=slice('1985-11-01', '2021-03-31')).where(data.time.dt.month.isin([11,12,1,2,3]), drop=True)
winter_months = data.sel(time=slice('1986-05-01', '2021-09-30')).where(data.time.dt.month.isin([5,6,7,8,9]), drop=True)

summer_series = summer_months.rolling(time=5, center=True).sum()[2::5]
winter_series = winter_months.rolling(time=5, center=True).sum()[2::5]

for i in range(df_n):
    
    print(i, df_n)
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
dfout.to_excel("../../../hyperdrought_data/forestv2/pr_data.xlsx", index=False)
##############

###############
# pr prev seas data 
dfout = pd.DataFrame(columns=df.columns, index=df.index)
ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
ds['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
data = ds['pr']
lat = data.lat
lon = data.lon

summer_months = data.sel(time=slice('1985-05-01', '2020-09-30')).where(data.time.dt.month.isin([5,6,7,8,9]), drop=True)
winter_months = data.sel(time=slice('1985-11-01', '2021-03-31')).where(data.time.dt.month.isin([11,12,1,2,3]), drop=True)

summer_series = summer_months.rolling(time=5, center=True).sum()[2::5]
winter_series = winter_months.rolling(time=5, center=True).sum()[2::5]

for i in range(df_n):
    
    print(i, df_n)
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
dfout.to_excel("../../../hyperdrought_data/forestv2/pr_prevseas_data.xlsx", index=False)
##############

###############
# pr year data 
dfout = pd.DataFrame(columns=df.columns, index=df.index)
ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
ds['time'] = pd.date_range('1960-01-01', '2021-12-31', freq='1MS')
data = ds['pr']
lat = data.lat
lon = data.lon

summer_months = data.sel(time=slice('1985-04-01', '2021-03-31'))
winter_months = data.sel(time=slice('1985-10-01', '2021-09-30'))

summer_series = summer_months.rolling(time=12, center=True).sum()[5::12]
winter_series = winter_months.rolling(time=12, center=True).sum()[5::12]

for i in range(df_n):
    
    print(i, df_n)
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
dfout.to_excel("../../../hyperdrought_data/forestv2/pr_annual_data.xlsx", index=False)
