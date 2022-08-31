import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm
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

precip_dry = precip.where(precip <= 80, drop = True)
nino34a_dry = nino34a.where(precip <= 80, drop = True)

nino43_dry_bef = nino34a_dry.where(nino34a_dry.time.dt.year < 1950, drop=True)
nino43_dry_aft = nino34a_dry.where(nino34a_dry.time.dt.year >= 1950, drop=True)

fit_bef = norm.fit(nino43_dry_bef.values)
fit_aft = norm.fit(nino43_dry_aft.values)

xx = np.linspace(-2, 2, 1000)
fig = plt.figure(figsize=(8,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.fill_between(xx, norm.pdf(xx, *fit_bef), where=xx<=norm.median(*fit_bef), color = 'b', alpha=0.2)
plt.fill_between(xx, norm.pdf(xx, *fit_aft), where=xx<=norm.median(*fit_aft), color = 'r', alpha=0.2)
plt.plot(xx, norm.pdf(xx, *fit_bef), color = 'b', label='Dry winters before 1950 - norm fit')
plt.plot(xx, norm.pdf(xx, *fit_aft), color = 'r', label='Dry winters after 1950 - norm fit')
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('MJJAS Niño3.4 index [ºC]')
plt.ylabel('PDF')
plt.legend(frameon=False)
# plt.xticks([0, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450])
# plt.xlim([-30, 450])
plt.ylim([0, 1.2])
plt.axvline(0, color='grey', lw=2, ls='--')
plt.axvline(-0.5, color='fuchsia', lw=2, ls='--')
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/enso_compare_precip_range.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()