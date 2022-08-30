import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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

mu_minus_1sigma = mu - sigma
mu_minus_2sigma = mu - 2*sigma

nboot = 10000
filepath = '../../../hyperdrought_data/output/MLE_precip_QN_GMST_'+str(nboot)+'.nc'
bspreds = xr.open_dataset(filepath)
bspreds_sigma0 = bspreds.sigma0.values
bspreds_eta = bspreds.eta.values
bspreds_alpha = bspreds.alpha.values

T2019 = smf.sel(time='2019').values
T1880 = smf.sel(time='1880').values

mu_2019 = mu.sel(time='2019')
mu_1880 = mu.sel(time='1880')

mu_2019_dist = bspreds_eta*bspreds_sigma0*np.exp(bspreds_alpha*T2019)
mu_1880_dist = bspreds_eta*bspreds_sigma0*np.exp(bspreds_alpha*T1880)

mu_2019_inf, mu_2019_sup = np.quantile(mu_2019_dist, [0.025, 0.975], axis = 0)
mu_1880_inf, mu_1880_sup = np.quantile(mu_1880_dist, [0.025, 0.975], axis = 0)

err_mu_2019_inf = mu_2019 - mu_2019_inf
err_mu_2019_sup = mu_2019_sup - mu_2019

err_mu_1880_inf = mu_1880 - mu_1880_inf
err_mu_1880_sup = mu_1880_sup - mu_1880

fig = plt.figure(figsize=(8,5))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.scatter(sm_no2019, qn_no2019, s=12, marker='o', edgecolor='blue', facecolor='none')
plt.scatter(smf.sel(time='2019'), qnf.sel(time='2019'), s=12, marker='s', edgecolor='fuchsia', facecolor='none')
plt.plot(sm, mu, color='red', linewidth = 2)
plt.plot(sm, mu_minus_1sigma, color='red', linewidth = 0.5)
plt.plot(sm, mu_minus_2sigma, color='red', linewidth = 0.5)

plt.errorbar(x=T1880, y=mu_1880, yerr=[err_mu_1880_inf, err_mu_1880_sup], lw=1.2, color='r', capsize=3, fmt = '.', capthick=1.5)
plt.errorbar(x=T2019, y=mu_2019, yerr=[err_mu_2019_inf, err_mu_2019_sup], lw=1.2, color='r', capsize=3, fmt = '.', capthick=1.5)

# plt.ylim([27, 33.75])
plt.grid(color='grey', lw=0.4, ls='--')
ax = plt.gca()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(direction="in")
plt.xlabel('Global mean surface temperature anomaly (smoothed) [ºC]')
plt.ylabel('Annual precip [mm]')
plt.savefig('../../../hyperdrought_data/png/MLE_precip_GMST_scatter.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()