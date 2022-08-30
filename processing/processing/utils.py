import xarray as xr
import pandas as pd
import numpy as np
from . import series as se
from . import cr2met as cr2
from scipy import stats
from scipy.stats import pearsonr

# get return periods from arg as numpy array
def get_return_periods(z, method='up'):
    
    n = z.size

    # sort values
    z = np.sort(z)

    # get unique values
    u = np.unique(z)

    m = u.size

    # create matrix for tail probability and tau
    tail = np.zeros((m,))
    tau = np.zeros((m,))

    # compute tail and tau
    for i in range(m):
        x = u[i]
        if method == 'down':
            tail[i] = np.sum(z<=x)/n
        else: # up
            tail[i] = np.sum(z>=x)/n
        tau[i] = 1/tail[i]

    return u, tau

# get corr between QN and CR2MET
def get_corr_QN_CR2MET():
    qn = se.get_QN_annual_precip_long_record()
    pr = cr2.get_cr2met_annual_precip()
    
    qn = qn.sel(time=slice('1979','2019'))
    pr = pr.sel(time=slice('1979','2019'))

    t, n, m = pr.values.shape
    matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            series = pr[:,i,j].values
            if np.isnan(series[0]):
                matrix[i,j] = np.nan
            else:
                r, p = pearsonr(series, qn.values)
                matrix[i,j] = r
    
    return xr.DataArray(matrix, coords=[pr.lat, pr.lon], dims=['lat', 'lon'])

# get corr between Puerto Montt and CR2MET
def get_corr_PM_CR2MET():
    qn = se.get_PM_JFM_precip()
    pr = cr2.get_cr2met_JFM_precip_acc()
    
    qn = qn.sel(time=slice('1979','2019'))
    pr = pr.sel(time=slice('1979','2019'))

    t, n, m = pr.values.shape
    matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            series = pr[:,i,j].values
            if np.isnan(series[0]):
                matrix[i,j] = np.nan
            else:
                r, p = pearsonr(series, qn.values)
                matrix[i,j] = r
    
    return xr.DataArray(matrix, coords=[pr.lat, pr.lon], dims=['lat', 'lon'])

# get corr between Coyhaique and CR2MET
def get_corr_CO_CR2MET():
    qn = se.get_CO_JFM_precip()
    pr = cr2.get_cr2met_JFM_precip_acc()
    
    qn = qn.sel(time=slice('1979','2019'))
    pr = pr.sel(time=slice('1979','2019'))

    t, n, m = pr.values.shape
    matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            series = pr[:,i,j].values
            if np.isnan(series[0]):
                matrix[i,j] = np.nan
            else:
                r, p = pearsonr(series, qn.values)
                matrix[i,j] = r
    
    return xr.DataArray(matrix, coords=[pr.lat, pr.lon], dims=['lat', 'lon'])


# get ranking 2019 annual precip
def get_CR2MET_2019_annual_precip_ranking():
    
    pr = cr2.get_cr2met_annual_precip()

    ntime, nlat, nlon = pr.values.shape
    matrix = np.zeros((nlat,nlon))

    for ilat in range(nlat):
        for ilon in range(nlon):
            series = pr[:, ilat, ilon].values
            if np.isnan(series[0]):
                matrix[ilat, ilon] = np.nan
            else:
                ranks = stats.rankdata(series)
                matrix[ilat, ilon] = ranks[-1]
    
    # to xarray DataArray
    da = xr.DataArray(matrix, coords = [pr.lat, pr.lon], dims = ['lat', 'lon'])
    return da

# get ranking 2016 DJF precip
def get_CR2MET_2016_JFM_precip_ranking():
    
    pr = cr2.get_cr2met_JFM_precip_acc()

    ntime, nlat, nlon = pr.values.shape
    matrix = np.zeros((nlat,nlon))

    for ilat in range(nlat):
        for ilon in range(nlon):
            series = pr[:, ilat, ilon].values
            if np.isnan(series[0]):
                matrix[ilat, ilon] = np.nan
            else:
                ranks = stats.rankdata(series)
                matrix[ilat, ilon] = ranks[-4]
    
    # to xarray DataArray
    da = xr.DataArray(matrix, coords = [pr.lat, pr.lon], dims = ['lat', 'lon'])
    return da