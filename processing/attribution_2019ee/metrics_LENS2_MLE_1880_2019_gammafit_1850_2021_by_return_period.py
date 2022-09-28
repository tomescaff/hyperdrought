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
lens2_prec_full = lens.get_LENS2_annual_precip_NOAA_QNEW()

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

df.loc['tau cf', 'raw'] = tau_cf
df.loc['tau ac', 'raw'] = tau_ac
df.loc['rr c-a', 'raw'] = rr_ca
df.loc['far c-a', 'raw'] = far_ca
df.loc['delta c-a', 'raw'] = delta

# bootstrap MLE
nboot = 100
filepath = '../../../hyperdrought_data/output/MLE_precip_LENS2_GMST_'+str(nboot)+'_evaluation.nc'
bspreds = xr.open_dataset(join(currentdir, filepath))
bspreds_sigma0 = bspreds.sigma0.values
bspreds_eta = bspreds.eta.values
bspreds_alpha = bspreds.alpha.values

Tac = lens2_gmst_full.sel(time = ac_year).values
Tcf = lens2_gmst_full.sel(time = cf_year).values

sig_ac_dist = bspreds_sigma0*np.exp(bspreds_alpha*Tac)
sig_cf_dist = bspreds_sigma0*np.exp(bspreds_alpha*Tcf)
eta_dist = bspreds_eta

bspreds_tau_cf = np.zeros((nboot,))
bspreds_tau_ac = np.zeros((nboot,))
bspreds_rr_ca = np.zeros((nboot,))
bspreds_far_ca = np.zeros((nboot,))
bspreds_delta = np.zeros((nboot,))

for i in range(nboot):

    tau_cf_i = 1/gamma.cdf(ev, eta_dist[i], 0, sig_cf_dist[i])
    tau_ac_i = 1/gamma.cdf(ev, eta_dist[i], 0, sig_ac_dist[i])

    rr_ca_i = tau_cf_i/tau_ac_i
    far_ca_i = (tau_cf_i-tau_ac_i)/tau_cf_i
    delta_i = ev - gamma.ppf(1/tau_ac_i, eta_dist[i], 0, sig_cf_dist[i])

    bspreds_tau_cf[i] = tau_cf_i
    bspreds_tau_ac[i] = tau_ac_i
    bspreds_rr_ca[i] = rr_ca_i
    bspreds_far_ca[i] = far_ca_i
    bspreds_delta[i] = delta_i

mapping = [('95ci lower', 0.025), ('95ci upper', 0.975), ('1percentile', 0.01)]

for col, thr in mapping:
    df.loc['tau cf', col] = np.quantile(bspreds_tau_cf, [thr], axis = 0)
    df.loc['tau ac', col] = np.quantile(bspreds_tau_ac, [thr], axis = 0)
    df.loc['rr c-a', col] = np.quantile(bspreds_rr_ca, [thr], axis = 0)
    df.loc['far c-a', col] = np.quantile(bspreds_far_ca, [thr], axis = 0)
    df.loc['delta c-a', col] = np.quantile(bspreds_delta, [thr], axis = 0)

df = df.applymap(lambda x: round(float(x),2))
df.to_csv(join(currentdir,f'../../../hyperdrought_data/output/metrics_LENS2_MLE_{cf_year}_{ac_year}_gammafit_1850_2021_by_return_period.csv'))