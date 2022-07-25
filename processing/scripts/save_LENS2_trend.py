import sys
import numpy as np
import xarray as xr
from scipy.stats import linregress

sys.path.append('..')

import processing.lens as lens

init_year = '1979'
end_year = '2019'

da = lens.get_LENS2_annual_precip()
da = da.sel(time=slice(init_year, end_year))

r, t, n, m = da.values.shape

matrix = np.zeros((r,n,m))

for k in range(r):
    for i in range(n):
        for j in range(m):
            series = da[k,:,i,j]
            x, y = series.time.dt.year.values, series.values
            slope, intercept, rvalue, pvalue, stderr = linregress(x, y)
            matrix[k,i,j] = slope

da_ans = xr.DataArray(matrix, coords=[da.run, da.lat, da.lon], dims=['run', 'lat', 'lon'])
ds_out = xr.Dataset({'slope':da_ans})
ds_out.to_netcdf('../../../hyperdrought_data/data/LENS2_trend_'+init_year+'_'+end_year+'.nc')


