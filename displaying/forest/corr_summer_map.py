import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
import matplotlib.ticker as mticker
import cmaps

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

df = pd.read_excel("../../../hyperdrought_data/forest/nbr_shp_19112022.dbf.xlsx")
n,m  = df.shape

data = df.iloc[:,4:]
columns = list(data.columns)

f = lambda x : True if x == '01' else False
summer = [f(col[-2:]) for col in columns]
winter = [not sum for sum in summer]  

nmax_summer = 2022-1986+1
nmax_winter = 2021-1986+1

no_nans_summer = np.zeros((n,))
no_nans_winter = np.zeros((n,))

full_data = []

for i in range(n):
    
    no_nans_summer[i] = np.count_nonzero(~np.isnan(data.iloc[i,summer].values))
    no_nans_winter[i] = np.count_nonzero(~np.isnan(data.iloc[i,winter].values))

    if nmax_summer - no_nans_summer[i] == 0:
        full_data = full_data + [i]

meta = df.iloc[full_data,:4]
data = data.iloc[full_data, summer]

df_pr = pd.read_excel("../../../hyperdrought_data/forest/precip_data.xlsx")
df_tmin = pd.read_excel("../../../hyperdrought_data/forest/tmin_data.xlsx")
df_tmax = pd.read_excel("../../../hyperdrought_data/forest/tmax_data.xlsx")

data_pr = df_pr.iloc[:,4:].iloc[full_data, summer]

data_tmin = df_tmin.iloc[:,4:].iloc[full_data, summer]
data_tmax = df_tmax.iloc[:,4:].iloc[full_data, summer]

nn, mm = data.shape

mat_pr = np.zeros((nn,))
mat_tmin = np.zeros((nn,))
mat_tmax = np.zeros((nn,))

for i in range(nn):
    mat_pr[i] = np.corrcoef(data.iloc[i,:-1].values, data_pr.iloc[i,:-1].values)[0,1]
    mat_tmin[i] = np.corrcoef(data.iloc[i,:-1].values, data_tmin.iloc[i,:-1].values)[0,1]
    mat_tmax[i] = np.corrcoef(data.iloc[i,:-1].values, data_tmax.iloc[i,:-1].values)[0,1]

ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
da = ds['pr']
winter_months = da.sel(time=slice('1985-07-01', '2022-10-31')).where(da.time.dt.month.isin([7,8,9,10]), drop=True)
winter_series = winter_months.rolling(time=4, center=True).sum()[2::4]
mat_pr_prev = np.zeros((nn,))

pr_prev = np.zeros((nn, mm))

for i in range(meta.shape[0]):
    row = meta.iloc[i,:]
    ws = winter_series.sel(lat=row['lat'], lon=row['long'], method='nearest')
    pr_prev[i, : ] = ws.values
    mat_pr_prev[i] = np.corrcoef(data.iloc[i,:-1].values, ws[:-1].values)[0,1]

da_tmin = ds['pr'][0,:,:].squeeze()*np.nan
da_tmax = da_tmin.copy()
da_prec = da_tmin.copy()
da_prev = da_tmin.copy()

for i in range(nn):
    
    alat = meta.iloc[i, 2]
    alon = meta.iloc[i, 3]
    ilon = list(ds.lon.values).index(ds.sel(lon=alon, method='nearest').lon)
    ilat = list(ds.lat.values).index(ds.sel(lat=alat, method='nearest').lat)

    da_tmin[ilat, ilon] = mat_tmin[i]
    da_tmax[ilat, ilon] = mat_tmax[i]
    da_prec[ilat, ilon] = mat_pr[i]
    da_prev[ilat, ilon] = mat_pr_prev[i]

fname = '../../../hyperdrought_data/shp/Regiones/Regional.shp'

da = da_prev

# create figure
fig = plt.figure(figsize=(8,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8


# define projection
ax = plt.axes(projection=ccrs.PlateCarree())

# set extent of map
ax.set_extent([-72, -70, -35, -31.5], crs=ccrs.PlateCarree())

# define and set  x and y ticks
xticks = [ -72, -71, -70]
yticks = [ -35, -34, -33, -32]
ax.set_xticks( xticks, crs=ccrs.PlateCarree())
ax.set_yticks( yticks, crs=ccrs.PlateCarree())

# format x and y labels
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

# add backgroung for land and ocean
resol = '50m'  
land = cfeature.NaturalEarthFeature('physical', 'land',  scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

ax.add_feature(land, linewidth=0.0, alpha=0.5)
ax.add_feature(ocean, alpha = 0.5)

# add grid using previous ticks
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='grey', alpha=0.7, linestyle='--', draw_labels=False)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)

# plot the std and reshape color bar 
pcm = ax.pcolormesh(da.lon.values, da.lat.values, da.values, cmap=cmaps.amwg_blueyellowred, zorder=4, vmin=-1, vmax=1, edgecolor='grey', lw=0.05)
cbar = plt.colorbar(pcm, aspect = 40, pad=0.03)

# draw the coastlines
#land = cfeature.NaturalEarthFeature('physical', 'land',  scale=resol, edgecolor='k', facecolor='none')
#ax.add_feature(land, linewidth=0.5, alpha=1, zorder=5)

ax.add_geometries(Reader(fname).geometries(), ccrs.Mercator.GOOGLE, facecolor='none', edgecolor='k', zorder=6, lw=0.4)

# reduce outline patch linewidths
cbar.outline.set_linewidth(0.4)
ax.spines['geo'].set_linewidth(0.4)

# circle = plt.Circle((-70.6828, -33.4450), 0.2, color='k', fill=False, zorder=4, lw=0.5)
# ax.add_patch(circle)

plt.savefig('../../../hyperdrought_data/forest/CR2MET_prev_corr_summer.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()













