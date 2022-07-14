import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import gamma
from scipy.stats import exponweib
from scipy.stats import invgamma
from scipy.stats import beta
from scipy.stats import lognorm
from scipy.stats import weibull_min

sys.path.append('../../processing')

import processing.lens as lens
import processing.series as se

# get Quinta Normal time series
qn = se.get_QN_annual_precip()
lens2 = lens.get_LENS2_annual_precip_QNEW()

# create figure
fig, axs = plt.subplots(5, 1, figsize=(10,7.5), constrained_layout=False, sharex=True)

# plt params
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8

# compute histogram
xmin, xmax, step = 0,700, 10
density = True
hist_qn, bins_ = np.histogram(qn.values, bins=np.arange(xmin, xmax+step, 50), density=density)
hist_lens2_full, bins = np.histogram(np.ravel(lens2.sel(time=slice('1950', '2021')).values), bins=np.arange(xmin, xmax+step, step), density=density)
hist_lens2_present, bins = np.histogram(np.ravel(lens2.sel(time=slice('1991', '2020')).values), bins=np.arange(xmin, xmax+step, step), density=density)
hist_lens2_past, bins = np.histogram(np.ravel(lens2.sel(time=slice('1851', '1880')).values), bins=np.arange(xmin, xmax+step, step), density=density)
hist_lens2_future, bins = np.histogram(np.ravel(lens2.sel(time=slice('2071', '2100')).values), bins=np.arange(xmin, xmax+step, step), density=density)

# plot the histogram
width = 0.9 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
width_ = 0.9 * (bins_[1] - bins_[0])
center_ = (bins_[:-1] + bins_[1:]) / 2
axs[0].bar(center_, hist_qn, align='center', width=width_, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'QN 1950-2021')
axs[1].bar(center, hist_lens2_full, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 1950-2021')
axs[2].bar(center, hist_lens2_present, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 actual (1991-2020)')
axs[3].bar(center, hist_lens2_past, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 counterfactual (1851-1880)')
axs[4].bar(center, hist_lens2_future, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 future (2071-2100)')

# # plot pdfs
x = np.linspace(xmin, xmax, 1000)
for dist, col, lab in [(gamma, 'k', 'gamma fit')]:
    axs[0].plot(x, dist.pdf(x, *dist.fit(qn.values)), lw=1.0, color=col, label=lab)
    axs[1].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('1950', '2021')).values))), lw=1.0, color=col, label=lab)
    axs[2].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('1991', '2020')).values))), lw=1.0, color=col, label=lab)
    axs[3].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('1851', '1880')).values))), lw=1.0, color=col, label=lab)
    axs[4].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('2071', '2100')).values))), lw=1.0, color=col, label=lab)

for ax in axs:
    plt.sca(ax)
    plt.grid(lw=0.2, ls='--', color='grey') # set grid
    plt.legend() # set legend
    # plt.xlim([xmin, xmax]) # set xlim
    plt.ylim([0, 0.005]) # set ylim
    plt.xlabel('Annual Precip (s.u.)') # set labels
    plt.ylabel('PDF')

plt.subplots_adjust(hspace=.0)
plt.savefig('../../../hyperdrought_data/png/LENS2_QN_distribution.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()