import sys
import numpy as np
import xarray as xr
from scipy.stats import linregress

sys.path.append('..')

import processing.cr2met as cr2met

init_year = '1979'
end_year = '2019'

da = cr2met.get_cr2met_ONDJFMA_precip() # mm/day per season
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
ds_out.to_netcdf('../../../hyperdrought_data/data/CR2MET_trend_ONDJFMA_'+init_year+'_'+end_year+'.nc')


