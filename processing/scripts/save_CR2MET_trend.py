import sys
import numpy as np
import xarray as xr
from scipy.stats import linregress

sys.path.append('..')

import processing.cr2met as cr2met

init_year = '1981'
end_year = '2010'

da = cr2met.get_cr2met_annual_precip()
da = da.sel(time=slice(init_year, end_year))

t, n, m = da.values.shape

matrix = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        series = da[:, i, j]
        x, y = series.time.dt.year.values, series.values
        slope, intercept, rvalue, pvalue, stderr = linregress(x, y)
        matrix[i,j] = slope

da_ans = xr.DataArray(matrix, coords=[da.lat, da.lon], dims=['lat', 'lon'])
ds_out = xr.Dataset({'slope':da_ans})
ds_out.to_netcdf('../../../hyperdrought_data/data/CR2MET_trend_'+init_year+'_'+end_year+'.nc')


