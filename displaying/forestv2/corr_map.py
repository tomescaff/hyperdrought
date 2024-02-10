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

df = pd.read_csv("../../../hyperdrought_data/forestv2/corr_nbr_pr_prevseas.csv", index_col=0)
df_index = pd.read_csv("latlon_index.csv", index_col=0)

data = df['r_02']
filename = 'CR2MET_corr_nbr_prevseas_winter.png'

n, m = df.shape
ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
da = ds['pr'][0,:,:].squeeze()*np.nan

for i in range(n):
    
    ilon, ilat = df_index.iloc[i, [4, 5]].astype(int)
    da[ilat, ilon] = data.iloc[i]

fname = '../../../hyperdrought_data/shp/Regiones/Regional.shp'

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

plt.savefig('../../../hyperdrought_data/forestv2/'+filename, dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()













