import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import gamma
from scipy.stats import exponweib
from scipy.stats import invgamma
from scipy.stats import beta
from scipy.stats import lognorm
from scipy.stats import weibull_min
from scipy.stats import pearson3

from scipy.optimize import curve_fit

sys.path.append('../../processing')

import processing.lens as lens
import processing.series as se

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()

# create figure
fig, axs = plt.subplots(5, 1, figsize=(10,7.5), constrained_layout=False, sharex=True)

# plt params
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8

# compute histogram
xmin, xmax, step = 0.1,700, 10
density = True
hist_qn, bins_ = np.histogram(qn.values, bins=np.arange(xmin, xmax+step, 40), density=density)
hist_lens2_full, bins = np.histogram(np.ravel(lens2.sel(time=slice('1866', '2021')).values), bins=np.arange(xmin, xmax+step, step), density=density)
hist_lens2_present, bins = np.histogram(np.ravel(lens2.sel(time=slice('1991', '2020')).values), bins=np.arange(xmin, xmax+step, step), density=density)
hist_lens2_past, bins = np.histogram(np.ravel(lens2.sel(time=slice('1851', '1880')).values), bins=np.arange(xmin, xmax+step, step), density=density)
hist_lens2_future, bins = np.histogram(np.ravel(lens2.sel(time=slice('2071', '2100')).values), bins=np.arange(xmin, xmax+step, step), density=density)

# compute centers
width_ = 0.9 * (bins_[1] - bins_[0])
center_ = (bins_[:-1] + bins_[1:]) / 2
width = 0.9 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

# compute LP3 parameters
def LP3(x, skew, loc, scale):
    return pearson3.pdf(np.log(x), skew, loc, scale)

p0 = [400, 0, 1]
popt_qn, _              = curve_fit(LP3, center_, hist_qn, p0=p0)
popt_lens2_full, _      = curve_fit(LP3, center, hist_lens2_full, p0=p0)
popt_lens2_present, _   = curve_fit(LP3, center, hist_lens2_present, p0=p0)
popt_lens2_past, _      = curve_fit(LP3, center, hist_lens2_past, p0=p0)
popt_lens2_future, _    = curve_fit(LP3, center, hist_lens2_future, p0=p0)

# plot the histogram
axs[0].bar(center_, hist_qn, align='center', width=width_, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'QN 1866-2021')
axs[1].bar(center, hist_lens2_full, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 1866-2021')
axs[2].bar(center, hist_lens2_present, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 actual (1991-2020)')
axs[3].bar(center, hist_lens2_past, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 counterfactual (1851-1880)')
axs[4].bar(center, hist_lens2_future, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'LENS2 future (2071-2100)')

# # plot pdfs
x = np.linspace(xmin, xmax, 1000)
for dist, col, lab in [(lognorm, 'k', 'LN2 fit'), (gamma, 'green', 'G2 fit')]:
    axs[0].plot(x, dist.pdf(x, *dist.fit(qn.values, floc=0, scale=1)), lw=1.0, color=col, label=lab)
    axs[1].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('1866', '2021')).values), floc=0, scale=1)), lw=1.0, color=col, label=lab)
    axs[2].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('1991', '2020')).values), floc=0, scale=1)), lw=1.0, color=col, label=lab)
    axs[3].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('1851', '1880')).values), floc=0, scale=1)), lw=1.0, color=col, label=lab)
    axs[4].plot(x, dist.pdf(x, *dist.fit(np.ravel(lens2.sel(time=slice('2071', '2100')).values), floc=0, scale=1)), lw=1.0, color=col, label=lab)

# dist, col, lab = LP3, 'r', 'LP3'
# axs[0].plot(x, dist(x, *popt_qn), lw=2.0, color=col, label=lab)
# axs[1].plot(x, dist(x, *popt_lens2_full), lw=2.0, color=col, label=lab)
# axs[2].plot(x, dist(x, *popt_lens2_present), lw=2.0, color=col, label=lab)
# axs[3].plot(x, dist(x, *popt_lens2_past), lw=2.0, color=col, label=lab)
# axs[4].plot(x, dist(x, *popt_lens2_future), lw=2.0, color=col, label=lab)

# axs[0].plot(x, dist.pdf(np.log(x), *dist.fit(np.log(qn.values))), lw=2.0, color=col, label=lab)
# axs[1].plot(x, dist.pdf(np.log(x), *dist.fit(np.log(np.ravel(lens2.sel(time=slice('1950', '2021')).values)), floc=0, fscale=1)), lw=2.0, color=col, label=lab)
# axs[2].plot(x, dist.pdf(np.log(x), *dist.fit(np.log(np.ravel(lens2.sel(time=slice('1991', '2020')).values)), floc=0, fscale=1)), lw=2.0, color=col, label=lab)
# axs[3].plot(x, dist.pdf(np.log(x), *dist.fit(np.log(np.ravel(lens2.sel(time=slice('1851', '1880')).values)), floc=0, fscale=1)), lw=2.0, color=col, label=lab)
# axs[4].plot(x, dist.pdf(np.log(x), *dist.fit(np.log(np.ravel(lens2.sel(time=slice('2071', '2100')).values)), floc=0, fscale=1)), lw=2.0, color=col, label=lab)

# plot mean values
col='r'
axs[0].axvline(np.mean(qn.values), lw=2., color=col, ls='--')
axs[1].axvline(np.mean(lens2.sel(time=slice('1866', '2021')).values), lw=2., color=col, ls='--')
axs[2].axvline(np.mean(lens2.sel(time=slice('1991', '2020')).values), lw=2., color=col, ls='--')
axs[3].axvline(np.mean(lens2.sel(time=slice('1851', '1880')).values), lw=2., color=col, ls='--')
axs[4].axvline(np.mean(lens2.sel(time=slice('2071', '2100')).values), lw=2., color=col, ls='--')


for ax in axs:
    plt.sca(ax)
    plt.grid(lw=0.2, ls='--', color='grey') # set grid
    plt.legend() # set legend
    plt.ylim([0, 0.005]) # set ylim
    plt.yticks(np.arange(0,0.005, 0.001))
    plt.xlabel('Annual Precip (mm)') # set labels
    plt.ylabel('PDF')

plt.subplots_adjust(hspace=.0)
plt.savefig('../../../hyperdrought_data/png/LENS2_QN_distribution.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()