import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
import matplotlib.ticker as mticker
import matplotlib

# read data
filepath = "../../../hyperdrought_data/forestv2/ndvi_20012023_metadata.xlsx"
df = pd.read_excel(filepath) # read excel
n,m  = df.shape
meta = df.iloc[:,:4] # get metadata section
data = df.iloc[:,15:] # get data section since 1992
columns = list(data.columns) # get data column names

# separate winter (02) from summer (01)
f = lambda x : True if x == '01' else False
mask_summer = [f(col[-2:]) for col in columns]
mask_winter = [not b for b in mask_summer]

# explore how many nans there are since 1992
nmax_summer = 2022-1992+1
nmax_winter = 2021-1992+1

valid_summer = np.zeros((n,))
valid_winter = np.zeros((n,))

summer_row_full_data = []
winter_row_full_data = []

# map of nans per grid cell
ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
da_summer = ds['pr'][0,:,:].squeeze()*np.nan
da_winter = da_summer.copy()
da_count = da_summer.copy()

for i in range(n):
    print(i, n)
    valid_summer[i] = np.count_nonzero(~np.isnan(data.iloc[i,mask_summer].values))
    valid_winter[i] = np.count_nonzero(~np.isnan(data.iloc[i,mask_winter].values))
    
    alat = meta.iloc[i, 2]
    alon = meta.iloc[i, 3]
    count = meta.iloc[i, 1]
    ilon = list(ds.lon.values).index(ds.sel(lon=alon, method='nearest').lon)
    ilat = list(ds.lat.values).index(ds.sel(lat=alat, method='nearest').lat)
    
    da_summer[ilat, ilon] = nmax_summer - valid_summer[i]
    da_winter[ilat, ilon] = nmax_winter - valid_winter[i]
    da_count[ilat, ilon] = count
    
    if  valid_summer[i] == nmax_summer:
        summer_row_full_data = summer_row_full_data + [i]  
    if  valid_winter[i] == nmax_winter:
        winter_row_full_data = winter_row_full_data + [i]

# save summer row full data
pd.DataFrame(summer_row_full_data).to_csv('summer_row_full_data.csv')

# plot number of nans per grid cell
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
fig, axs = plt.subplots(1,2, figsize=(12,7))
plt.sca(axs[0])
plt.hist(nmax_summer - valid_summer, bins = np.arange(-0.5, 75, 0.5)) 
plt.title('Summer')
plt.xlabel('Nan values')
plt.ylabel('Frequency')
plt.sca(axs[1])
plt.hist(nmax_winter - valid_winter, bins = np.arange(-0.5, 75, 0.5))
plt.title('Winter')
plt.xlabel('Nan values')
plt.ylabel('Frenquency')
filepath = "../../../hyperdrought_data/forestv2/ndvi_nan_histogram.png"
plt.savefig(filepath, dpi=300)
plt.show()

# plot map of nans
for data_array, name in zip([da_summer, da_winter], ['summer_nans', 'winter_nans']):
    da = data_array
    fig = plt.figure(figsize=(8,7)) # create figure
    plt.rcParams["font.family"] = 'Arial'
    plt.rcParams["font.size"] = 8
    fname = '../../../hyperdrought_data/shp/Regiones/Regional.shp'
    ax = plt.axes(projection=ccrs.PlateCarree()) # define projection
    ax.set_extent([-72, -70, -35, -31.5], crs=ccrs.PlateCarree()) # set extent of map
    xticks = [ -72, -71, -70] # define and set x and y ticks
    yticks = [ -35, -34, -33, -32]
    ax.set_xticks( xticks, crs=ccrs.PlateCarree())
    ax.set_yticks( yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True) # format x and y labels
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    resol = '50m'  # add backgroung for land and ocean
    land = cfeature.NaturalEarthFeature('physical', 'land',  scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    ax.add_feature(land, linewidth=0.0, alpha=0.5)
    ax.add_feature(ocean, alpha = 0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='grey', alpha=0.7, linestyle='--', draw_labels=False)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    pcm = ax.pcolormesh(da.lon.values, da.lat.values, da.values, cmap='jet', zorder=4, vmin=0, vmax=10, edgecolor='grey', lw=0.05)
    cbar = plt.colorbar(pcm, aspect = 40, pad=0.03)
    ax.add_geometries(Reader(fname).geometries(), ccrs.Mercator.GOOGLE, facecolor='none', edgecolor='k', zorder=6, lw=0.4)
    cbar.outline.set_linewidth(0.4) # reduce outline patch linewidths
    ax.spines['geo'].set_linewidth(0.4)
    filepath = f'../../../hyperdrought_data/forestv2/map_{name}.png'
    plt.savefig(filepath, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

# TODO: plot map of count in log scale from 113 to 22983

da = da_count
fig = plt.figure(figsize=(8,7)) # create figure
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8
fname = '../../../hyperdrought_data/shp/Regiones/Regional.shp'
ax = plt.axes(projection=ccrs.PlateCarree()) # define projection
ax.set_extent([-72, -70, -35, -31.5], crs=ccrs.PlateCarree()) # set extent of map
xticks = [ -72, -71, -70] # define and set x and y ticks
yticks = [ -35, -34, -33, -32]
ax.set_xticks( xticks, crs=ccrs.PlateCarree())
ax.set_yticks( yticks, crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True) # format x and y labels
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
resol = '50m'  # add backgroung for land and ocean
land = cfeature.NaturalEarthFeature('physical', 'land',  scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
ax.add_feature(land, linewidth=0.0, alpha=0.5)
ax.add_feature(ocean, alpha = 0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='grey', alpha=0.7, linestyle='--', draw_labels=False)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)
pcm = ax.pcolormesh(da.lon.values, da.lat.values, da.values, cmap='jet', zorder=4, vmin=113, vmax=22983, norm=matplotlib.colors.LogNorm(), edgecolor='grey', lw=0.05)
cbar = plt.colorbar(pcm, aspect = 40, pad=0.03)
ax.add_geometries(Reader(fname).geometries(), ccrs.Mercator.GOOGLE, facecolor='none', edgecolor='k', zorder=6, lw=0.4)
cbar.outline.set_linewidth(0.4) # reduce outline patch linewidths
ax.spines['geo'].set_linewidth(0.4)
filepath = f'../../../hyperdrought_data/forestv2/map_count.png'
plt.savefig(filepath, dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
