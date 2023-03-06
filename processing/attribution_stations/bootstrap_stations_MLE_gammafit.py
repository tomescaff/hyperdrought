import sys
import numpy as np
import xarray as xr
from sklearn.utils import resample as bootstrap
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.math as pmath


smf = gmst.get_gmst_annual_5year_smooth()

filepath = '../../../hyperdrought_data/CR2_explorer/cr2_pr_stns_data_before_1990_24deg_40degS.nc'
stnsf = xr.open_dataset(join(currentdir, filepath))

N = stnsf.stn.size

for j in range(N):
    prf = stnsf.data[:, j]
    name = str(stnsf.name[j].values)

    prev_year = int(xr.where(np.isnan(prf), 1,np.nan).dropna('time').time.dt.year[-1].values)
    pr = prf.sel(time=slice(str(prev_year+1), '2021'))
    sm = smf.sel(time=slice(str(prev_year+1), '2021'))
    
    sm = sm.where(sm.time.dt.year != 2019, drop=True)
    pr = pr.where(pr.time.dt.year != 2019, drop=True)

    # # bootstrap
    nboot = 10
    bspreds_sigma0 = np.zeros((nboot,))
    bspreds_eta = np.zeros((nboot,))
    bspreds_alpha = np.zeros((nboot,))

    for i in range(nboot):
        pr_i, sm_i = bootstrap(pr.values, sm.values)
        xopt_i = pmath.mle_gamma_2d_fast(pr_i, sm_i, [70, 4, -0.5])
        bspreds_sigma0[i] = xopt_i[0]
        bspreds_eta[i] = xopt_i[1]
        bspreds_alpha[i] = xopt_i[2]

    iter = np.arange(nboot)
    ds = xr.Dataset({
        'sigma0': xr.DataArray(bspreds_sigma0, coords=[iter], dims=['iter']),
        'eta':    xr.DataArray(bspreds_eta,    coords=[iter], dims=['iter']),
        'alpha':  xr.DataArray(bspreds_alpha,  coords=[iter], dims=['iter']), 
    })
    filepath = f'../../../hyperdrought_data/output/MLE_2019ee_GMST_nboot_{nboot}_gamma_stn_{j:02}.nc'
    ds.to_netcdf(join(currentdir, filepath))