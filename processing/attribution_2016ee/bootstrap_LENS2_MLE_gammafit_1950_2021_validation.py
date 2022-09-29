import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.utils import resample as bootstrap
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.lens as lens
import processing.math as pmath
import processing.series as se

lens2_gmst_full = gmst.get_gmst_annual_lens2_ensmean()
lens2_prec_full = lens.get_LENS2_JFM_precip_NOAA_PM()

lens2_gmst = lens2_gmst_full.sel(time=slice('1950', '2021'))
lens2_prec = lens2_prec_full.sel(time=slice('1950', '2021'))

lens2_prec_norm = lens2_prec/lens2_prec.mean('time')

lens2_gmst_arr = np.tile(lens2_gmst.values, lens2_prec_norm.shape[0])
lens2_prec_arr = np.ravel(lens2_prec_norm.values)

# # bootstrap
nboot = 10
bspreds_sigma0 = np.zeros((nboot,))
bspreds_eta = np.zeros((nboot,))
bspreds_alpha = np.zeros((nboot,))

for i in range(nboot):
    qn_i, sm_i = bootstrap(lens2_prec_arr, lens2_gmst_arr)
    xopt_i = pmath.mle_gamma_2d_fast(qn_i, sm_i, [0.09, 10.83, -0.09])
    bspreds_sigma0[i] = xopt_i[0]
    bspreds_eta[i] = xopt_i[1]
    bspreds_alpha[i] = xopt_i[2]

iter = np.arange(nboot)
ds = xr.Dataset({
    'sigma0': xr.DataArray(bspreds_sigma0, coords=[iter], dims=['iter']),
    'eta':    xr.DataArray(bspreds_eta,    coords=[iter], dims=['iter']),
    'alpha':  xr.DataArray(bspreds_alpha,  coords=[iter], dims=['iter']), 
})
filepath = '../../../hyperdrought_data/output/PM_MLE_precip_LENS2_GMST_'+str(nboot)+'_validation.nc'
ds.to_netcdf(join(currentdir,filepath))