import sys
import numpy as np
from scipy.stats import linregress
from scipy.stats import t
from scipy.stats import lognorm
from scipy.stats import gamma
from sklearn.utils import resample as bootstrap
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.gmst as gmst
import processing.lens as lens
import processing.utils as ut
import processing.series as se

lens2_gmst_full = gmst.get_gmst_annual_lens2_ensmean()
lens2_prec_full = lens.get_LENS2_JFM_precip_NOAA_PM_NN()

#lens2_prec_full = lens2_prec_full.rolling(time=1).mean().dropna('time')

lens2_gmst = lens2_gmst_full.sel(time=slice('1850', '2021'))
lens2_prec = lens2_prec_full.sel(time=slice('1850', '1880'))

data = np.ravel(lens2_prec.values)

cbins = np.arange(0, 600, 20)
xx = np.linspace(0, 600, 1000)
hist_2017, bins = np.histogram(data, bins=cbins, density=True)
width = 1.0 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2


fig = plt.figure(figsize=(13,8))
plt.bar(center, hist_2017, align='center', width=width, edgecolor='k', facecolor='b', color='blue', alpha = 0.25, label='2017 LENS2')
plt.plot(xx, gamma.pdf(xx, *gamma.fit(data, floc=0, scale=1)), color='k', label='gamma')
plt.plot(xx, lognorm.pdf(xx, *lognorm.fit(data, floc=0, scale=1)), color='r', label='lognorm')
plt.show()


xx = np.linspace(0.1, 600, 1000)
gammafit = gamma.fit(data, floc=0, scale=1)
y_gamma = 1/gamma.cdf(xx, *gammafit)
lnfit = lognorm.fit(data, floc=0, scale=1)
y_ln = 1/lognorm.cdf(xx, *lnfit)
u, tau = ut.get_return_periods(data, method='down')


fig = plt.figure(figsize=(12,6))
plt.plot(y_gamma, xx, color='k', lw=0.8, alpha = 1, label = 'Parametric return period (gamma)')
plt.plot(y_ln, xx, color='r', lw=0.8, alpha = 1, label = 'Parametric return period (LN2)')
plt.scatter(tau, u, marker='o', facecolor='lightskyblue', edgecolor='blue', color='blue', alpha = 1, label = 'Non parametric return period')
# plot the max value line
plt.axhline(np.max(data), lw=1, color='grey', ls='dotted', label='max value')
plt.grid(lw=0.2, ls='--', color='grey')
plt.legend()
plt.gca().set_xscale('log')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
plt.xlim([0.9,11000])
plt.ylim([0, 600])
plt.xlabel('Return period (years)')
plt.ylabel('Precip (mm)')
plt.show()
