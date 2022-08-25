import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import exponweib
from scipy.stats import invgamma
from scipy.stats import beta
from scipy.stats import lognorm
from scipy.stats import weibull_min
from scipy.stats import pearson3
from sklearn.utils import resample as bootstrap

sys.path.append('../../processing')

import processing.series as se
import processing.lens as lens 
import processing.utils as ut

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()
qn_1866_2021 = qn.sel(time=slice('1866', '2021'))
qn_mean_1866_2021 = qn_1866_2021.mean('time')
qn_anom = qn.sel(time='2019') - qn.sel(time=slice('1991', '2020')).mean('time')

# get LENS2 time series ensamble
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()
lens2_mean_1866_2021 = lens2.sel(time=slice('1866', '2021')).mean('time')
lens2_anom = lens2 - lens2_mean_1866_2021
lens2_fixed_1866_2021 = lens2_anom + qn_mean_1866_2021

# get past period
da_jan_1851_1880 = lens2_fixed_1866_2021.sel(time=slice('1851','1880'))
np_jan_all_runs_1851_1880 = np.ravel(da_jan_1851_1880.values)

# get 1991-2020 period
da_jan_1991_2020 = lens2_fixed_1866_2021.sel(time=slice('1991', '2020'))
np_jan_all_runs_1991_2020 = np.ravel(da_jan_1991_2020.values)

# get 2071-2100 period
da_jan_2071_2100 = lens2_fixed_1866_2021.sel(time=slice('2071', '2100'))
np_jan_all_runs_2071_2100 = np.ravel(da_jan_2071_2100.values)

# fitting distributions
g2fit_1851_1880 = gamma.fit(np_jan_all_runs_1851_1880, floc=0)
g2fit_1991_2020 = gamma.fit(np_jan_all_runs_1991_2020, floc=0)
g2fit_2071_2100 = gamma.fit(np_jan_all_runs_2071_2100, floc=0)

# computing best x ticks
y = np.linspace(40, 850, 1000)
x = 1/gamma.cdf(y, *g2fit_2071_2100)

# computing y values
y_g2_1851_1880 = gamma.ppf(1/x, *g2fit_1851_1880)
y_g2_1991_2020 = gamma.ppf(1/x, *g2fit_1991_2020)
y_g2_2071_2100 = gamma.ppf(1/x, *g2fit_2071_2100)

#confidence intervals
nboot = 1000

# bootstraping LENS 1851_1880
bspreds = np.zeros((nboot, x.size))
for i in range(nboot):
    z = bootstrap(np_jan_all_runs_1851_1880)
    g2fit_ = gamma.fit(z, floc=0, scale=1)
    bspreds[i] = gamma.ppf(1/x, *g2fit_)
yinf_1851_1880, ysup_1851_1880 = np.quantile(bspreds, [0.025, 0.975], axis = 0)

# bootstraping LENS 1991-2020
bspreds = np.zeros((nboot, x.size))
for i in range(nboot):
    z = bootstrap(np_jan_all_runs_1991_2020)
    g2fit_ = gamma.fit(z, floc=0, scale=1)
    bspreds[i] = gamma.ppf(1/x, *g2fit_)
yinf_1991_2020, ysup_1991_2020 = np.quantile(bspreds, [0.025, 0.975], axis = 0)

# bootstraping LENS 2071-2100
bspreds = np.zeros((nboot, x.size))
for i in range(nboot):
    z = bootstrap(np_jan_all_runs_2071_2100)
    g2fit_ = gamma.fit(z, floc=0, scale=1)
    bspreds[i] = gamma.ppf(1/x, *g2fit_)
yinf_2071_2100, ysup_2071_2100 = np.quantile(bspreds, [0.025, 0.975], axis = 0)

# get anom line
ev = qn_anom + np.mean(np_jan_all_runs_1991_2020)
tau = 1/gamma.cdf(ev, *g2fit_1991_2020)

past_ev = gamma.ppf(1/tau, *g2fit_1851_1880)
future_ev = gamma.ppf(1/tau, *g2fit_2071_2100)

tau_ee_1851_1880 = 1/gamma.cdf(ev, *g2fit_1851_1880)
tau_ee_2071_2100 = 1/gamma.cdf(ev, *g2fit_2071_2100)

# create figure
fig = plt.figure(figsize=(8,6))

# plt params
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

# plot the confidence intervals
plt.gca().fill_between(x, yinf_1851_1880, ysup_1851_1880, color='b', alpha=.25)
plt.gca().fill_between(x, yinf_1991_2020, ysup_1991_2020, color='k', alpha=.25)
plt.gca().fill_between(x, yinf_2071_2100, ysup_2071_2100, color='r', alpha=.25)

# plot the paramtric curves
plt.plot(x, y_g2_1851_1880, color='b', lw=1.5, alpha = 1, label = 'Parametric return period PAST', zorder=4)
plt.plot(x, y_g2_1991_2020, color='k', lw=1.5, alpha = 1, label = 'Parametric return period PRESENT', zorder=4)
plt.plot(x, y_g2_2071_2100, color='r', lw=1.5, alpha = 1, label = 'Parametric return period FUTURE', zorder=4)

# plot the ev 
plt.axhline(ev, lw=1, color='grey', ls='-')

# plot the mean values
plt.axhline(np.mean(np_jan_all_runs_1851_1880), lw=1, color='b', ls='dotted')
plt.axhline(np.mean(np_jan_all_runs_1991_2020), lw=1, color='k', ls='dotted')
plt.axhline(np.mean(np_jan_all_runs_2071_2100), lw=1, color='r', ls='dotted')

# plot the ev value
plt.axhline(past_ev, lw=1, color='b', ls='--')
plt.axhline(future_ev, lw=1, color='r', ls='--')

# plot ee tau value
plt.axvline(tau, lw=1, color='k', ls='dotted')
plt.axvline(tau_ee_1851_1880, lw=1, color='b', ls='--')
plt.axvline(tau_ee_2071_2100, lw=1, color='r', ls='--')

# set grid
plt.grid(lw=0.2, ls='--', color='grey')

# set legend
plt.gca().set_xscale('log')

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
plt.xlim([0.9,500])
plt.ylim([0,850])
# set title and labels
plt.xlabel('Return period (years)')
plt.ylabel('Annual precip (mm)')
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig('../../../hyperdrought_data/png/QN_return_period_ci_g2.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()