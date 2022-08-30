import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.utils import resample as bootstrap
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.series as se
import processing.math as pmath

smf = gmst.get_gmst_annual_5year_smooth()
qnf = se.get_QN_annual_precip_long_record()

sm = smf.sel(time=slice('1880','2021'))
qn = qnf.sel(time=slice('1880','2021'))

sm = sm.where(sm.time.dt.year != 2019, drop=True)
qn = qn.where(qn.time.dt.year != 2019, drop=True)

# # bootstrap
nboot = 10000
bspreds_sigma0 = np.zeros((nboot,))
bspreds_eta = np.zeros((nboot,))
bspreds_alpha = np.zeros((nboot,))

for i in range(nboot):
    qn_i, sm_i = bootstrap(qn.values, sm.values)
    xopt_i = pmath.mle_gamma_2d(qn_i, sm_i, [70, 4, -0.5])
    bspreds_sigma0[i] = xopt_i[0]
    bspreds_eta[i] = xopt_i[1]
    bspreds_alpha[i] = xopt_i[2]

iter = np.arange(nboot)
ds = xr.Dataset({
    'sigma0': xr.DataArray(bspreds_sigma0, coords=[iter], dims=['iter']),
    'eta':    xr.DataArray(bspreds_eta,    coords=[iter], dims=['iter']),
    'alpha':  xr.DataArray(bspreds_alpha,  coords=[iter], dims=['iter']), 
})
filepath = '../../../hyperdrought_data/output/MLE_precip_QN_GMST_'+str(nboot)+'.nc'
ds.to_netcdf(join(currentdir,filepath))