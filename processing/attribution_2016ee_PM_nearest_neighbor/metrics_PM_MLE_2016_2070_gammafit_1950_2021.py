import sys
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import gamma
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.series as se
import processing.math as pmath
import processing.cr2met_v25 as cr2met

index = ['tau ac', 'tau fu', 'rr a-f', 'far a-f', 'delta a-f']
columns = ['raw', '95ci lower', '95ci upper', '1percentile']
df = pd.DataFrame(columns=columns, index=index)

ac_year = '2016'
fu_year = '2070'

lens1_gmst_full = gmst.get_gmst_annual_lens1_ensmean()
T2070 = lens1_gmst_full.sel(time='2070').values

# raw values

smf = gmst.get_gmst_annual_5year_smooth()
qnf = se.get_PM_JFM_precip()

sm = smf.sel(time=slice('1950','2021'))
qn = qnf.sel(time=slice('1950','2021'))

sm_no2019 = sm.where(sm.time.dt.year != 2016, drop=True)
qn_no2019 = qn.where(qn.time.dt.year != 2016, drop=True)

xopt = pmath.mle_gamma_2d(qn_no2019.values, sm_no2019.values, [70, 4, -0.5])

sigma0, eta, alpha = xopt
sigma = sigma0*np.exp(alpha*smf)

sig_MLE_ac = sigma.sel(time = ac_year)
sig_MLE_fu = sigma0*np.exp(alpha*T2070)
eta_MLE = eta

# get ev value 
ev = qn.sel(time='2016')

# get return periods
tau_fu = 1/gamma.cdf(ev, eta, 0, sig_MLE_fu)
tau_ac = 1/gamma.cdf(ev, eta, 0, sig_MLE_ac)

# get rr and far
rr_af = tau_ac/tau_fu
far_af = (tau_ac-tau_fu)/tau_ac

# get delta
ev2 = gamma.ppf(1/tau_ac, eta, 0, sig_MLE_fu)
delta = 100*(ev2 - ev)/ev 

df.loc['tau fu', 'raw'] = tau_fu
df.loc['tau ac', 'raw'] = tau_ac
df.loc['rr a-f', 'raw'] = rr_af
df.loc['far a-f', 'raw'] = far_af
df.loc['delta a-f', 'raw'] = delta

# bootstrap MLE
nboot = 10
filepath = '../../../hyperdrought_data/output/PM_MLE_precip_PM_GMST_'+str(nboot)+'_evaluation.nc'
bspreds = xr.open_dataset(join(currentdir, filepath))
bspreds_sigma0 = bspreds.sigma0.values
bspreds_eta = bspreds.eta.values
bspreds_alpha = bspreds.alpha.values

Tac = smf.sel(time = ac_year).values
Tcf = T2070

sig_ac_dist = bspreds_sigma0*np.exp(bspreds_alpha*Tac)
sig_fu_dist = bspreds_sigma0*np.exp(bspreds_alpha*Tcf)
eta_dist = bspreds_eta

bspreds_tau_fu = np.zeros((nboot,))
bspreds_tau_ac = np.zeros((nboot,))
bspreds_rr_af = np.zeros((nboot,))
bspreds_far_af = np.zeros((nboot,))
bspreds_delta = np.zeros((nboot,))

for i in range(nboot):

    tau_fu_i = 1/gamma.cdf(ev, eta_dist[i], 0, sig_fu_dist[i])
    tau_ac_i = 1/gamma.cdf(ev, eta_dist[i], 0, sig_ac_dist[i])

    rr_af_i = tau_ac_i/tau_fu_i
    far_af_i = (tau_ac_i-tau_fu_i)/tau_ac_i
    ev2_i = gamma.ppf(1/tau_ac_i, eta_dist[i], 0, sig_fu_dist[i])
    delta_i = 100*(ev2_i - ev)/ev

    bspreds_tau_fu[i] = tau_fu_i
    bspreds_tau_ac[i] = tau_ac_i
    bspreds_rr_af[i] = rr_af_i
    bspreds_far_af[i] = far_af_i
    bspreds_delta[i] = delta_i

mapping = [('95ci lower', 0.025), ('95ci upper', 0.975), ('1percentile', 0.01)]

for col, thr in mapping:
    df.loc['tau fu', col] = np.quantile(bspreds_tau_fu, [thr], axis = 0)
    df.loc['tau ac', col] = np.quantile(bspreds_tau_ac, [thr], axis = 0)
    df.loc['rr a-f', col] = np.quantile(bspreds_rr_af, [thr], axis = 0)
    df.loc['far a-f', col] = np.quantile(bspreds_far_af, [thr], axis = 0)
    df.loc['delta a-f', col] = np.quantile(bspreds_delta, [thr], axis = 0)

df = df.applymap(lambda x: round(float(x),2))
df.to_csv(join(currentdir,f'../../../hyperdrought_data/output/metrics_PM_MLE_{ac_year}_{fu_year}_gammafit_1950_2021.csv'))