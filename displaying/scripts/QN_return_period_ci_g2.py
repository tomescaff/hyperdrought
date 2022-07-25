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
import processing.utils as ut

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()

# get Quinta Normal return period
u_do, tau_do = ut.get_return_periods(qn.values, method='down')

# fitting distributions
qn_g2fit = gamma.fit(np.ravel(qn.values), floc=0, scale=1)
qn_ln2fit = lognorm.fit(np.ravel(qn.values), floc=0, scale=1)

# computing best x ticks
y = np.linspace(40, 850, 1000)
x = 1/gamma.cdf(y, *qn_g2fit)

# computing y values
y_qn_g2 = gamma.ppf(1/x, *qn_g2fit)
y_qn_ln2 = lognorm.ppf(1/x, *qn_ln2fit)

#confidence intervals
nboot = 1000

# bootstraping Gamma2
bspreds = np.zeros((nboot, x.size))
for i in range(nboot):
    z = bootstrap(qn.values)
    g2fit_ = gamma.fit(z, floc=0, scale=1)
    bspreds[i] = gamma.ppf(1/x, *g2fit_)
yinf_g2, ysup_g2 = np.quantile(bspreds, [0.025, 0.975], axis = 0)

# bootstraping LN2
bspreds = np.zeros((nboot, x.size))
for i in range(nboot):
    z = bootstrap(qn.values)
    ln2fit_ = lognorm.fit(z, floc=0, scale=1)
    bspreds[i] = lognorm.ppf(1/x, *ln2fit_)
yinf_ln2, ysup_ln2 = np.quantile(bspreds, [0.025, 0.975], axis = 0)

# create figure
fig = plt.figure(figsize=(8,6))

# plt params
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8

# plot the confidence intervals
plt.gca().fill_between(x, yinf_g2, ysup_g2, color='b', alpha=.25)
# plt.gca().fill_between(x, yinf_ln2, ysup_ln2, color='r', alpha=.25)

# plot the scatter
plt.scatter(tau_do, u_do, marker='o', facecolor='grey', edgecolor='grey', color='grey', alpha = 0.5, label = 'Non parametric return period')

# plot the paramtric curves
plt.plot(x, y_qn_g2, color='b', lw=1.5, alpha = 1, label = 'Parametric return period G2')
# plt.plot(x, y_qn_ln2, color='r', lw=1.5, alpha = 1, label = 'Parametric return period LN2')

# plot the 1866-2021 clim
plt.axhline(qn.mean(), lw=1, color='grey', ls='--', label='1866-2021 mean value')

# plot the 2019 line
ev = qn.sel(time='2019').values
plt.axhline(ev, lw=1, color='grey', ls='dotted', label='2019 value')

# print taus

tau_g2 = float(1/gamma.cdf(ev, *qn_g2fit))
# tau_ln2 = float(1/lognorm.cdf(ev, *qn_ln2fit))
print(f'tau G2: {tau_g2:.2f}')
# print(f'tau LN2: {tau_ln2:.2f}')


# set grid
plt.grid(lw=0.2, ls='--', color='grey')

# set legend
plt.legend()
plt.gca().set_xscale('log')
# plt.gca().set_yscale('log')

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
plt.xlim([0.9,500])
plt.ylim([0,850])
# set title and labels
plt.xlabel('Return period (years)')
plt.ylabel('Annual precip (mm)')
plt.title('Annual precip return period at Quinta Normal')
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig('../../../hyperdrought_data/png/QN_return_period_ci_g2.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()