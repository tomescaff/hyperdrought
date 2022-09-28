import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se
import processing.lens as lens

# get Quinta Normal time series
pm = se.get_PM_JFM_precip().sel(time=slice('1950', '2021'))

# get LENS1 time series
lens1 = lens.get_LENS1_JFM_precip_NOAA_PM()
lens1_sup = lens1.mean('run') + lens1.std('run')
lens1_inf = lens1.mean('run') - lens1.std('run')

# get LENS2 time series
lens2 = lens.get_LENS2_JFM_precip_NOAA_PM()
lens2_sup = lens2.mean('run') + lens2.std('run')
lens2_inf = lens2.mean('run') - lens2.std('run')

# compute normalized values
pm_norm = pm/pm.sel(time=slice('1980','2010')).mean('time')*100
lens1_norm = lens1/lens1.sel(time=slice('1980','2010')).mean('time')*100
lens2_norm = lens2/lens2.sel(time=slice('1980','2010')).mean('time')*100

lens1_norm_sup = lens1_norm.mean('run') + lens1_norm.std('run')
lens1_norm_inf = lens1_norm.mean('run') - lens1_norm.std('run')

lens2_norm_sup = lens2_norm.mean('run') + lens2_norm.std('run')
lens2_norm_inf = lens2_norm.mean('run') - lens2_norm.std('run')

fig, axs = plt.subplots(3,2, figsize=(15,8))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

plt.sca(axs[0,0])
plt.plot(lens1.time.dt.year, lens1.values.T, 'skyblue', lw=1)
plt.plot(lens1.time.dt.year, lens1.mean('run'), 'b', label='PM-lens1')
plt.plot(pm.time.dt.year, pm.values, 'r', lw=2, label='PM-obs')
plt.xlim([1840, 2100])
plt.ylim([0, 800])
plt.ylabel('JFM precip. (mm)')
plt.legend(loc='upper left')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")

plt.sca(axs[1,0])
plt.plot(lens2.time.dt.year, lens2.values.T, 'grey', lw=1)
plt.plot(lens2.time.dt.year, lens2.mean('run'), 'k', label='PM-lens2')
plt.plot(pm.time.dt.year, pm.values, 'r', lw=2, label='PM-obs')
plt.xlim([1840, 2100])
plt.ylim([0, 800])
plt.ylabel('JFM precip. (mm)')
plt.legend(loc='upper left')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")

plt.sca(axs[2,0])
plt.fill_between(lens1.time.dt.year, lens1_inf, lens1_sup, color='b', lw=1, alpha=0.3, label='PM-lens1 mu+-std')
plt.fill_between(lens2.time.dt.year, lens2_inf, lens2_sup, color='k', lw=1, alpha=0.3, label='PM-lens2 mu+-std')
plt.plot(lens1.time.dt.year, lens1.mean('run'), 'b', label='PM-lens1')
plt.plot(lens2.time.dt.year, lens2.mean('run'), 'k', label='PM-lens2')
plt.plot(pm.time.dt.year, pm.values, 'r', lw=2, label='QN-obs')
plt.xlim([1840, 2100])
plt.ylim([0, 800])
plt.legend(loc='upper left')
plt.ylabel('JFM precip. (mm)')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")

plt.sca(axs[0,1])
plt.plot(lens1_norm.time.dt.year, lens1_norm.values.T, 'skyblue', lw=1)
plt.plot(lens1_norm.time.dt.year, lens1_norm.mean('run'), 'b')
plt.plot(pm_norm.time.dt.year, pm_norm.values, 'r', lw=2)
plt.xlim([1840, 2100])
plt.ylim([0, 300])
plt.ylabel('Norm. JFM precip. (%)')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")

plt.sca(axs[1,1])
plt.plot(lens2_norm.time.dt.year, lens2_norm.values.T, 'grey', lw=1)
plt.plot(lens2_norm.time.dt.year, lens2_norm.mean('run'), 'k')
plt.plot(pm_norm.time.dt.year, pm_norm.values, 'r', lw=2)
plt.xlim([1840, 2100])
plt.ylim([0, 300])
plt.ylabel('Norm. JFM precip. (%)')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")

plt.sca(axs[2,1])
plt.fill_between(lens1_norm.time.dt.year, lens1_norm_inf, lens1_norm_sup, color='b', lw=1, alpha=0.3)
plt.fill_between(lens2_norm.time.dt.year, lens2_norm_inf, lens2_norm_sup, color='k', lw=1, alpha=0.3)
plt.plot(lens1_norm.time.dt.year, lens1_norm.mean('run'), 'b')
plt.plot(lens2_norm.time.dt.year, lens2_norm.mean('run'), 'k')
plt.plot(pm_norm.time.dt.year, pm_norm.values, 'r', lw=2)
plt.xlim([1840, 2100])
plt.ylim([0, 300])
plt.ylabel('Norm. JFM precip (%)')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")

plt.tight_layout()
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/PM_LENS1_LENS2_time_series_validation.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()