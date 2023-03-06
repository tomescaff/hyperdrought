import sys
import cmaps
import xarray as xr
import matplotlib
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

import processing.cr2met_v25 as cr2met

# get CR2MET precip
taupath = '../../../hyperdrought_data/output/CR2METv25_2019_return_period.nc'
da = xr.open_dataset(join(currentdir, taupath))['tau']
mask = cr2met.get_cl_mask()
da = da*mask

lats = [-33.45, -34.97, -36.59, -36.78,-39.65,  -33.66]
lons = [-70.68, -71.22, -72.04, -73.06, -73.08,-71.61]
retp = [5.4e1,   1.9e2, 4.7,    3.5,   2.8e1,  3.9e1]


fname = '../../../hyperdrought_data/shp/Regiones/Regional.shp'

# create figure
fig = plt.figure(figsize=(8,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8


# define projection
ax = plt.axes(projection=ccrs.PlateCarree())

# set extent of map
ax.set_extent([-74.5, -68, -40, -24], crs=ccrs.PlateCarree())

# define and set  x and y ticks
xticks = [ -74, -72, -70, -68]
yticks = [ -40, -38, -36, -34, -32, -30, -28, -26, -24]
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
pcm = ax.pcolormesh(da.lon.values, da.lat.values, da.values, cmap=cmaps.WhiteBlueGreenYellowRed, zorder=4, vmin=1, vmax=1e6, norm=matplotlib.colors.LogNorm())
cbar = plt.colorbar(pcm, aspect = 40, pad=0.03)

# plot the stations
ax.scatter(lons, lats, s=10, c=retp, cmap=cmaps.WhiteBlueGreenYellowRed, zorder=5, edgecolors='k', norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e6))

# draw the coastlines
#land = cfeature.NaturalEarthFeature('physical', 'land',  scale=resol, edgecolor='k', facecolor='none')
#ax.add_feature(land, linewidth=0.5, alpha=1, zorder=5)

ax.add_geometries(Reader(join(currentdir, fname)).geometries(), ccrs.Mercator.GOOGLE, facecolor='none', edgecolor='k', zorder=6, lw=0.4)

# reduce outline patch linewidths
cbar.outline.set_linewidth(0.4)
ax.spines['geo'].set_linewidth(0.4)

# circle = plt.Circle((-70.6828, -33.4450), 0.2, color='k', fill=False, zorder=4, lw=0.5)
# ax.add_patch(circle)

plt.savefig(join(currentdir,'../../../hyperdrought_data/png/CR2METv25_central_chile_precip_1960_2021_return_period_2019_with_stations.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()