import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gamma
from sklearn.utils import resample as bootstrap

sys.path.append('../../processing')

import processing.gmst as gmst
import processing.series as se
import processing.math as pmath

smf = gmst.get_gmst_annual_5year_smooth()
qnf = se.get_QN_annual_precip_long_record()

sm = smf.sel(time=slice('1880','2021'))
qn = qnf.sel(time=slice('1880','2021'))

sm_no2019 = sm.where(sm.time.dt.year != 2019, drop=True)
qn_no2019 = qn.where(qn.time.dt.year != 2019, drop=True)

xopt = pmath.mle_gamma_2d(qn_no2019.values, sm_no2019.values, [70, 4, -0.5])

sigma0, eta, alpha = xopt
sigma = sigma0*np.exp(alpha*sm)
mu = eta*sigma

mu_MLE_2019 = mu.sel(time='2019')
mu_MLE_1880 = mu.sel(time='1880')

qn_30yr_2019 = qnf.sel(time=slice('1992','2021'))
qn_30yr_1880 = qnf.sel(time=slice('1866','1895'))
eta_30yr_2019, loc_30yr_2019, sigma_30yr_2019 = gamma.fit(qn_30yr_2019, floc=0, scale=1)
eta_30yr_1880, loc_30yr_1880, sigma_30yr_1880 = gamma.fit(qn_30yr_1880, floc=0, scale=1)
mu_30yr_2019 = eta_30yr_2019*sigma_30yr_2019 
mu_30yr_1880 = eta_30yr_1880*sigma_30yr_1880

# bootstrap MLE
nboot = 10000
filepath = '../../../hyperdrought_data/output/MLE_precip_QN_GMST_'+str(nboot)+'.nc'
bspreds = xr.open_dataset(filepath)
bspreds_sigma0 = bspreds.sigma0.values
bspreds_eta = bspreds.eta.values
bspreds_alpha = bspreds.alpha.values

T2019 = smf.sel(time='2019').values
T1880 = smf.sel(time='1880').values

mu_2019_dist = bspreds_eta*bspreds_sigma0*np.exp(bspreds_alpha*T2019)
mu_1880_dist = bspreds_eta*bspreds_sigma0*np.exp(bspreds_alpha*T1880)

mu_mle_2019_inf, mu_mle_2019_sup = np.quantile(mu_2019_dist, [0.025, 0.975], axis = 0)
mu_mle_1880_inf, mu_mle_1880_sup = np.quantile(mu_1880_dist, [0.025, 0.975], axis = 0)

# bootstrap 30yr
bspreds_mu_2019 = np.zeros((nboot,))
bspreds_mu_1880 = np.zeros((nboot,))
for i in range(nboot):
    z2019_i = bootstrap(qnf.sel(time=slice('1992','2021')).values)
    z1880_i = bootstrap(qnf.sel(time=slice('1866','1895')).values)
    eta_2019_i, loc_2019_i, sigma_2019_i = gamma.fit(z2019_i, floc=0, scale=1) 
    eta_1880_i, loc_1880_i, sigma_1880_i = gamma.fit(z1880_i, floc=0, scale=1)
    bspreds_mu_2019[i] = eta_2019_i*sigma_2019_i
    bspreds_mu_1880[i] = eta_1880_i*sigma_1880_i

mu_30y_2019_inf, mu_30y_2019_sup = np.quantile(bspreds_mu_2019, [0.025, 0.975], axis = 0)
mu_30y_1880_inf, mu_30y_1880_sup = np.quantile(bspreds_mu_1880, [0.025, 0.975], axis = 0)

fig = plt.figure(figsize=(12,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.plot(qnf.time.dt.year, qnf.values, color='grey', lw=0.8, marker='.', ls='--')
plt.plot(mu.time.dt.year, mu.values, color='r', lw=0.8)
qn_roll = qn.rolling(time=30, min_periods=30, center=True).mean('time').dropna('time')
plt.plot(qn_roll.time.dt.year, qn_roll.values, color='b', lw=0.8)
plt.errorbar(x=1880+0.2, y=mu_MLE_1880.values, yerr=[mu_MLE_1880-mu_mle_1880_inf, mu_mle_1880_sup-mu_MLE_1880], lw=1.2, color='r', capsize=6, fmt = '.', capthick=1.5)
plt.errorbar(x=1880-0.2, y=mu_30yr_1880, yerr=[[mu_30yr_1880-mu_30y_1880_inf], [mu_30y_1880_sup-mu_30yr_1880]], lw=1.2, color='b', capsize=6, fmt = '.', capthick=1.5)
plt.errorbar(x=2019+0.2, y=mu_MLE_2019.values, yerr=[mu_MLE_2019-mu_mle_2019_inf, mu_mle_2019_sup-mu_MLE_2019], lw=1.2, color='r', capsize=6, fmt = '.', capthick=1.5)
plt.errorbar(x=2019-0.2, y=mu_30yr_2019, yerr=[[mu_30yr_2019-mu_30y_2019_inf], [mu_30y_2019_sup-mu_30yr_2019]], lw=1.2, color='b', capsize=6, fmt = '.', capthick=1.5)
plt.xlabel('Time')
plt.ylabel('Annual precip (mm)')
plt.legend(['QN annual precip', 'mu=mu(t) -- MLE', 'mu=mu(t) -- 30yr rolling average', 'mu 95%CI -- MLE', 'mu 95%CI -- 30yr'], frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig('../../../hyperdrought_data/png/comparison_MLE_30yr_mean.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
