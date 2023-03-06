import pandas as pd
import numpy as np
import xarray as xr

def daily_acc_with_nans(x, axis, **kwargs):
    frac=0.9
    cnt = lambda col: np.count_nonzero(~np.isnan(col))
    red = lambda col : np.nan if cnt(col)/col.size < frac else np.nansum(col)
    ans = np.apply_along_axis(red, axis, x)
    return ans

stnfile = '../../../hyperdrought_data/CR2_explorer/cr2_prDaily_2022.nc'

ds = xr.open_dataset(stnfile)
ds = ds.where( (ds.lat <= -24) & (ds.lat >= -40), drop=True)

pr = ds['pr']
lat = ds['lat']
lon = ds['lon']
alt = ds['alt']
name = ds['name']

pr = pr.where(pr != -9999.0)
pr_mon = pr.resample(time='1MS').reduce(daily_acc_with_nans, dim='time')
pr_ann = pr_mon.resample(time='1YS').sum('time', skipna = False)


# for the norm anom map
clim = pr_ann.sel(time=slice('1991', '2020')).mean('time')
anom = pr_ann.sel(time='2019')/clim*100-100
anom = anom.dropna('stn')
anom_data = anom.squeeze()
anom_lats = lat.sel(stn=anom_data.stn)
anom_lons = lon.sel(stn=anom_data.stn)
anom_alts = alt.sel(stn=anom_data.stn)
anom_name = name.sel(stn=anom_data.stn)

ds_anom = xr.Dataset({'data':anom_data, 'lat':anom_lats, 'lon':anom_lons, 'alt':anom_alts, 'name':anom_name})
ds_anom.to_netcdf('../../../hyperdrought_data/CR2_explorer/cr2_pr_norm_anom2019_refperiod_1991_2020_24deg_40degS.nc', encoding={"name": {"dtype": "str"}})

# for the std anom map
clim_std = pr_ann.sel(time=slice('1991', '2020')).std('time')
anom = (pr_ann.sel(time='2019')-clim)/clim_std
anom = anom.dropna('stn')
anom_data = anom.squeeze()
anom_lats = lat.sel(stn=anom_data.stn)
anom_lons = lon.sel(stn=anom_data.stn)
anom_alts = alt.sel(stn=anom_data.stn)
anom_name = name.sel(stn=anom_data.stn)

ds_anom = xr.Dataset({'data':anom_data, 'lat':anom_lats, 'lon':anom_lons, 'alt':anom_alts, 'name':anom_name})
ds_anom.to_netcdf('../../../hyperdrought_data/CR2_explorer/cr2_pr_std_anom2019_refperiod_1991_2020_24deg_40degS.nc', encoding={"name": {"dtype": "str"}})


# fro the return period map
mat = np.zeros((pr_ann.stn.size,))
for i in range(pr_ann.stn.size):
    k = 0
    for j in range(pr_ann.time.size):
        if np.isnan(pr_ann[-(j+1),i]):
            break
        else:
            k=k+1
    mat[i] = k

mask = mat>31
long_record_data = pr_ann[:, mask]
long_record_lats = lat[mask]
long_record_lons = lon[mask]
long_record_alts = alt[mask]
long_record_name = name[mask]
long_record_mat = mat[mask]

ds_long_record = xr.Dataset({'data':long_record_data, 'lat':long_record_lats, 'lon':long_record_lons, 'alt':long_record_alts, 'name':long_record_name, 'nblock':long_record_mat})
ds_long_record.to_netcdf('../../../hyperdrought_data/CR2_explorer/cr2_pr_stns_data_before_1990_24deg_40degS.nc', encoding={"name": {"dtype": "str"}})
