import matplotlib.pyplot as plt

import sys

sys.path.append('../../processing')

import processing.series as se
import processing.utils as ut
import numpy as np

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()
qn_rm2019 = qn.where(qn.time.dt.year != 2019, drop=True)

# get Quinta Normal return period
u_do, tau_do = ut.get_return_periods(qn.values, method='down')
u_do_rm2019, tau_do_rm2019 = ut.get_return_periods(qn_rm2019.values, method='down')

# create figure
fig = plt.figure(figsize=(8,6))

# plt params
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8

# plot the scatter
plt.scatter(tau_do, u_do, marker='o', facecolor='lightskyblue', edgecolor='blue', color='blue', alpha = 1, label = 'Non parametric return period (1866-2021)')

# plot the scatter
plt.scatter(tau_do_rm2019, u_do_rm2019, marker='o', facecolor='lightcoral', edgecolor='red', color='red', alpha = 1, label = 'Non parametric return period (2019 removed)')

# plot the 1866-2021 clim
plt.axhline(qn.mean(), lw=1, color='grey', ls='--', label='1866-2021 mean value')

# plot the 2019 line
plt.axhline(qn.sel(time='2019').values, lw=1, color='grey', ls='dotted', label='2019 value')


# set grid
plt.grid(lw=0.2, ls='--', color='grey')

# set legend
plt.legend()
plt.gca().set_xscale('log')
# plt.gca().set_yscale('log')

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])
plt.xlim([0.9,200])
# set title and labels
plt.xlabel('Return period (years)')
plt.ylabel('Annual precip (mm)')
plt.title('Annual precip return period at Quinta Normal')
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig('../../../hyperdrought_data/png/QN_non_parametric_return_period.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()