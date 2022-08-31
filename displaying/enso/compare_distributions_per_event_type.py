import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import gamma
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se
import processing.utils as ut

nino34a = se.get_MJJAS_Nino34_long_record()
precip = se.get_QN_MJJAS_precip_long_record()

nino34a = nino34a.sel(time=slice('1870', '2021'))
precip = precip.sel(time=slice('1870', '2021'))

precip = precip/precip.sel(time=slice('1981', '2010')).mean('time')*100

years_eln = precip.time.dt.year.where(nino34a >= 0.5, drop=True)
years_neu = precip.time.dt.year.where((nino34a > -0.5) & (nino34a < 0.5), drop=True)
years_lan = precip.time.dt.year.where(nino34a <= -0.5, drop=True)

precip_eln = precip.where(precip.time.dt.year.isin(years_eln), drop=True)
precip_neu = precip.where(precip.time.dt.year.isin(years_neu), drop=True)
precip_lan = precip.where(precip.time.dt.year.isin(years_lan), drop=True)

fit_eln = gamma.fit(precip_eln.values, floc=0, scale=1)
fit_neu = gamma.fit(precip_neu.values, floc=0, scale=1)
fit_lan = gamma.fit(precip_lan.values, floc=0, scale=1)

x_min = 0
x_max = 450
xx = np.linspace(x_min, x_max, 1000)
fig = plt.figure(figsize=(8,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.fill_between(xx, gamma.pdf(xx, *fit_lan), where=xx<=gamma.median(*fit_lan), color = 'b', alpha=0.2)
plt.fill_between(xx, gamma.pdf(xx, *fit_neu), where=xx<=gamma.median(*fit_neu), color = 'k', alpha=0.2)
plt.fill_between(xx, gamma.pdf(xx, *fit_eln), where=xx<=gamma.median(*fit_eln), color = 'r', alpha=0.2)
plt.plot(xx, gamma.pdf(xx, *fit_lan), color = 'b', label='La Niña - gamma fit (N=38)')
plt.plot(xx, gamma.pdf(xx, *fit_neu), color = 'k', label='Neutral - gamma fit (N=97)')
plt.plot(xx, gamma.pdf(xx, *fit_eln), color = 'r', label='El Niño - gamma fit (N=17)')
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('Quinta Normal MJJAS norm precipitation [%]')
plt.ylabel('PDF')
plt.legend(frameon=False)
plt.xticks([0, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450])
plt.xlim([-30, 450])
plt.ylim([0, 0.014])
plt.axvline(100, color='grey', lw=2, ls='--')
plt.axvline(80, color='fuchsia', lw=2, ls='--')
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/enso_compare_dist.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

precip_neu_bef1950 = precip_neu.where(precip_neu.time.dt.year < 1950, drop=True)
precip_neu_aft1950 = precip_neu.where(precip_neu.time.dt.year >= 1950, drop=True)

fit_bef = gamma.fit(precip_neu_bef1950.values, floc=0, scale=1)
fit_aft = gamma.fit(precip_neu_aft1950.values, floc=0, scale=1)

fig = plt.figure(figsize=(8,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.fill_between(xx, gamma.pdf(xx, *fit_bef), where=xx<=gamma.median(*fit_bef), color = 'b', alpha=0.2)
plt.fill_between(xx, gamma.pdf(xx, *fit_aft), where=xx<=gamma.median(*fit_aft), color = 'r', alpha=0.2)
plt.plot(xx, gamma.pdf(xx, *fit_bef), color = 'b', label='Neutral before 1950 - gamma fit (N=53)')
plt.plot(xx, gamma.pdf(xx, *fit_aft), color = 'r', label='Neutral after 1950 - gamma fit (N=44)')
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('Quinta Normal MJJAS norm precipitation [%]')
plt.ylabel('PDF')
plt.legend(frameon=False)
plt.xticks([0, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450])
plt.xlim([-30, 450])
plt.ylim([0, 0.014])
plt.axvline(100, color='grey', lw=2, ls='--')
plt.axvline(80, color='fuchsia', lw=2, ls='--')
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/enso_compare_dist_neutral.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()