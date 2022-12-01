import sys
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import gamma
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.lens as lens
import processing.math as pmath
import processing.series as se

index = ['tau cf', 'tau ac', 'rr c-a', 'far c-a', 'delta c-a']
columns = ['raw', '95ci lower', '95ci upper', '1percentile']
df = pd.DataFrame(columns=columns, index=index)

ac_year = '2019'
cf_year = '1880'

# raw values

lens2_gmst_full = gmst.get_gmst_annual_lens2_ensmean()
lens2_prec_full = lens.get_LENS2_annual_precip_NOAA_QN_NN()

lens2_gmst = lens2_gmst_full.sel(time=slice('1850', '2021'))
lens2_prec = lens2_prec_full.sel(time=slice('1850', '2021'))

lens2_gmst_arr = np.tile(lens2_gmst.values, lens2_prec.shape[0])
lens2_prec_arr = np.ravel(lens2_prec.values)

xopt = pmath.mle_gamma_2d(lens2_prec_arr, lens2_gmst_arr, [70, 4, -0.5])

sigma0, eta, alpha = xopt
sigma = sigma0*np.exp(alpha*lens2_gmst_full)

sig_MLE_ac = sigma.sel(time = ac_year)
sig_MLE_cf = sigma.sel(time = cf_year)
eta_MLE = eta

# get ev value 

tau_ac = 54

ev = gamma.ppf(1/tau_ac, eta, 0, sig_MLE_ac)

# get return periods
tau_cf = 1/gamma.cdf(ev, eta, 0, sig_MLE_cf)
tau_ac = 1/gamma.cdf(ev, eta, 0, sig_MLE_ac)

# get rr and far
rr_ca = tau_cf/tau_ac
far_ca = (tau_cf-tau_ac)/tau_cf

# get delta
delta = ev - gamma.ppf(1/tau_ac, eta, 0, sig_MLE_cf)
mean_2019 = gamma.stats(eta, 0, sig_MLE_ac, 'm')
df.loc['tau cf', 'raw'] = tau_cf
df.loc['tau ac', 'raw'] = tau_ac
df.loc['rr c-a', 'raw'] = rr_ca
df.loc['far c-a', 'raw'] = far_ca
df.loc['delta c-a', 'raw'] = delta



df = df.applymap(lambda x: round(float(x),2))
# df.to_csv(join(currentdir,f'../../../hyperdrought_data/output/metrics_LENS2_MLE_{cf_year}_{ac_year}_gammafit_1850_2021_by_return_period_QN_NN.csv'))