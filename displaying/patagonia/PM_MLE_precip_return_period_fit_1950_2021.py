import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.utils import resample as bootstrap
from os.path import join, abspath, dirname
from scipy.stats import gamma

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.gmst as gmst
import processing.utils as ut
import processing.math as pmath
import processing.cr2met_v25 as cr2met
import processing.series as se


smf = gmst.get_gmst_annual_5year_smooth()
qnf = se.get_PM_JFM_precip()

sm = smf.sel(time=slice('1950','2021'))
qn = qnf.sel(time=slice('1950','2021'))

sm_no2019 = sm.where(sm.time.dt.year != 2016, drop=True)
qn_no2019 = qn.where(qn.time.dt.year != 2016, drop=True)

xopt = pmath.mle_gamma_2d(qn_no2019.values, sm_no2019.values, [70, 4, -0.5])

sigma0, eta, alpha = xopt
sigma = sigma0*np.exp(alpha*sm)
mu = eta*sigma

nboot = 10
filepath = '../../../hyperdrought_data/output/PM_MLE_precip_PM_GMST_'+str(nboot)+'_evaluation.nc'
bspreds = xr.open_dataset(join(currentdir, filepath))
bspreds_sigma0 = bspreds.sigma0.values
bspreds_eta = bspreds.eta.values
bspreds_alpha = bspreds.alpha.values

T2019 = smf.sel(time='2016').values
T1880 = smf.sel(time='1880').values

mu_1880 = eta*sigma0*np.exp(alpha*T1880)
mu_2019 = eta*sigma0*np.exp(alpha*T2019)

sigma_1880 = sigma0*np.exp(alpha*T1880)
sigma_2019 = sigma0*np.exp(alpha*T2019)

mu_2019_dist = bspreds_eta*bspreds_sigma0*np.exp(bspreds_alpha*T2019)
mu_1880_dist = bspreds_eta*bspreds_sigma0*np.exp(bspreds_alpha*T1880)

eta_dist = bspreds_eta
sigma_1880_dist = bspreds_sigma0*np.exp(bspreds_alpha*T1880)
sigma_2019_dist = bspreds_sigma0*np.exp(bspreds_alpha*T2019)

# TODO: terminar grafico de periodos de retorno
# lo de abajo es de otro script, borrar 

y = np.linspace(40, 1200, 1000)

x_norm_2019 = 1/gamma.cdf(y, eta, 0, sigma_2019)
x_norm_1880 = 1/gamma.cdf(y, eta, 0, sigma_1880)

matrix_2019 = np.zeros((nboot, y.size))
matrix_1880 = np.zeros((nboot, y.size))

for i in range(nboot):
    matrix_2019[i,:] = 1/gamma.cdf(y, eta_dist[i], 0, sigma_2019_dist[i])
    matrix_1880[i,:] = 1/gamma.cdf(y, eta_dist[i], 0, sigma_1880_dist[i])

xinf_2019, xsup_2019= np.quantile(matrix_2019, [0.025, 0.975], axis = 0)
xinf_1880, xsup_1880= np.quantile(matrix_1880, [0.025, 0.975], axis = 0)

# get Quinta Normal return period
qn_shift_2019 = qn/mu*mu_2019
qn_shift_1880 = qn/mu*mu_1880

u_shift_2019, tau_shift_2019 = ut.get_return_periods(qn_shift_2019.values, method='down')
u_shift_1880, tau_shift_1880 = ut.get_return_periods(qn_shift_1880.values, method='down')

fig = plt.figure(figsize=(8,5))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

# plot the paramtric curves
plt.plot(x_norm_2019, y, color='r', lw=1.2, label = 'Gamma scale fit 2016')
plt.plot(x_norm_1880, y, color='b', lw=1.2, label = 'Gamma scale fit 1880')

plt.plot(xinf_2019[xinf_2019 > 10], y[xinf_2019 > 10], color='r', lw=0.5)
plt.plot(xsup_2019[xsup_2019 > 10], y[xsup_2019 > 10], color='r', lw=0.5)
plt.plot(xinf_1880[xinf_1880 > 10], y[xinf_1880 > 10], color='b', lw=0.5)
plt.plot(xsup_1880[xsup_1880 > 10], y[xsup_1880 > 10], color='b', lw=0.5)

plt.scatter(tau_shift_2019, u_shift_2019, marker='+', color='r', s=40, lw=0.5)
plt.scatter(tau_shift_1880, u_shift_1880, marker='x', color='b', s=40, lw=0.5)

plt.axhline(qnf.sel(time='2016'), color='fuchsia', lw=1.2, label='Observed 2016')

# set grid
plt.grid(lw=0.2, ls='--', color='grey')

# set legend
plt.legend(loc='upper right')

# set x log scale
plt.gca().set_xscale('log')

# set ticks and lims
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
plt.xlim([0.9,11000])
plt.ylim([0,600])

# set title and labels
ax=plt.gca()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(direction="in")
plt.xlabel('Return period (yr)')
plt.ylabel('JFM Precip (mm)')
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/PM_MLE_precip_return_period_fit_1950_2021.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()