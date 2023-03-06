import xarray as xr
import pandas as pd
import numpy as np
from . import series as se
from . import cr2met as cr2
from . import gmst
from . import math as pmath
from . import cr2met_v25 as cr2v25
from scipy import stats
from scipy.stats import pearsonr, gamma

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

def get_nearest_90p_contour(x, y, x_min, x_max, y_min, y_max):
    X = np.vstack((x,y)).T
    sigma = np.cov(X.T)
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    def fun(x0,x1):
        x = np.array([x0,x1])
        const=1/(2*np.pi*det)**0.5
        arg = -(x-X.mean(axis=0)).T@inv@(x-X.mean(axis=0))
        return np.exp(arg)*const
    def d2(x0,x1):
        x = np.array([x0,x1])
        return (x-X.mean(axis=0)).T@inv@(x-X.mean(axis=0))
    funvec = np.vectorize(fun)
    d2vec = np.vectorize(d2)
    x_ = np.linspace(x_min,x_max,100)
    y_ = np.linspace(y_min,y_max,100)
    xx,yy = np.meshgrid(x_,y_)
    zz = funvec(xx,yy)
    dd_pres = d2vec(xx,yy)
    return xx, yy, dd_pres

def get_return_period_cr2met():
    
    cr2met_data = cr2v25.get_cr2met_annual_precip()
    mask = cr2v25.get_cl_mask()
    cr2met_data = cr2met_data*mask
    t,n,m = cr2met_data.shape
    matrix = np.zeros((n,m))

    sm = gmst.get_gmst_annual_5year_smooth().sel(time=slice('1960', '2021'))
    sm_no2019 = sm.where(sm.time.dt.year != 2019, drop=True)

    for i in range(n):
        for j in range(m):
            if np.isnan(mask[i,j].values):
                matrix[i,j] = np.nan
            else:
                series = cr2met_data[:, i,j]
                series_n02019 = series.where(series.time.dt.year != 2019, drop=True)
                xopt = pmath.mle_gamma_2d_fast(series_n02019.values, sm_no2019.values, [70, 4, -0.5])
                sigma0, eta, alpha = xopt
                sigma = sigma0*np.exp(alpha*sm)
                
                sig_MLE_ac = sigma.sel(time = '2019')
                eta_MLE = eta

                # get ev value 
                ev = series.sel(time='2019')

                # get return periods
                tau_ac = 1/gamma.cdf(ev, eta_MLE, 0, sig_MLE_ac)
                matrix[i,j] = tau_ac
    da = xr.DataArray(matrix, coords = [cr2met_data.lat, cr2met_data.lon], dims=['lat', 'lon'])
    return da