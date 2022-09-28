import sys
import cmaps
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.lens as lens

# get LENS2 precip
da = lens.get_LENS2_JFM_precip_NOAA()
da = da.sel(time=slice('1979', '2019')).mean('time').mean('run')/(31+28+31)
fname = join(currentdir, '../../../hyperdrought_data/shp/Regiones/Regional.shp')

# create figure
fig = plt.figure(figsize=(8,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8

# define projection
ax = plt.axes(projection=ccrs.PlateCarree())

# set extent of map
ax.set_extent([-76, -66.5, -56.5, -17], crs=ccrs.PlateCarree())

# define and set  x and y ticks
xticks = [ -76, -72, -68]
yticks = [ -55, -50, -45, -40, -35, -30, -25, -20]
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

# plot the climatology and reshape color bar 
pcm = ax.pcolormesh(da.lon.values, da.lat.values, da.values, cmap=cmaps.amwg_blueyellowred, zorder=4, vmin=0, vmax=8, edgecolor='k', lw=0.05)
cbar = plt.colorbar(pcm, aspect = 40, pad=0.03)

# draw the coastlines
land = cfeature.NaturalEarthFeature('physical', 'land',  scale=resol, edgecolor='k', facecolor='none')
ax.add_feature(land, linewidth=0.5, alpha=1, zorder=5)

ax.add_geometries(Reader(fname).geometries(), ccrs.Mercator.GOOGLE, facecolor='none', edgecolor='k', zorder=6, lw=0.4)

#  reduce outline patch linewidths
cbar.outline.set_linewidth(0.4)
ax.spines['geo'].set_linewidth(0.4)

circle = plt.Circle((-70.6828, -33.4450), 0.2, color='k', fill=False, zorder=4, lw=0.5)
circle_pm = plt.Circle((-73.095833, -41.447499), 0.2, color='k', fill=False, zorder=4, lw=0.5)
# lon: 286.2, 287.5
# lat: -41.937173, -40.994764

ax.add_patch(circle)
ax.add_patch(circle_pm)

plt.savefig(join(currentdir, '../../../hyperdrought_data/png/LENS2_precip_JFM_1979_2019_clim.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()