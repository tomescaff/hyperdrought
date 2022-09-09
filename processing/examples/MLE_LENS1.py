import sys
import numpy as np

sys.path.append('..')

import processing.gmst as gmst
import processing.lens as lens
import processing.math as pmath
import processing.series as se

lens1_gmst_full = gmst.get_gmst_annual_lens1_ensmean()
lens1_prec_full = lens.get_LENS1_annual_precip_NOAA_QNEW()

# qn_full = se.get_QN_annual_precip_long_record()
# qn = qn_full.sel(time=slice('1930', '2021'))

lens1_gmst = lens1_gmst_full.sel(time=slice('1930', '2021'))
lens1_prec = lens1_prec_full.sel(time=slice('1930', '2021'))

lens1_prec_norm = lens1_prec/lens1_prec.mean('time')

lens1_gmst_arr = np.tile(lens1_gmst.values, lens1_prec_norm.shape[0])
lens1_prec_arr = np.ravel(lens1_prec_norm.values)

xarr = lens1_prec_arr
Tarr = lens1_gmst_arr
init_params = [0.09, 10.83, -0.09]

xopt = pmath.mle_gamma_2d_fast(xarr, Tarr, init_params)
print(xopt)