import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se

pm = se.get_PM_mon_precip()
mat = np.reshape(pm.values, (-1, 12))

pm_2015 = pm.sel(time=slice('2015-01', '2015-12'))
pm_2016 = pm.sel(time=slice('2016-01', '2016-12'))

fig, axs = plt.subplots(2,1, figsize=(10,7))
plt.rcParams["font.family"] = 'Arial'

plt.sca(axs[0])
plt.boxplot(mat, notch=False, meanline=True,showmeans=True, positions=np.arange(1,13), patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='green'), meanprops=dict(color='brown'))
plt.scatter(pm_2015.time.dt.month, pm_2015.values, color='b', label='2015 values')
plt.scatter(pm_2016.time.dt.month, pm_2016.values, color='r', label='2016 values')
plt.xlim([0,13])
plt.xticks(np.arange(1,13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
plt.legend()
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")

plt.sca(axs[1])
plt.bar(np.arange(1,13)-0.125, pm_2015 - np.mean(mat, axis=0), width=0.25,  color='b', label='2015 mon anom')
plt.bar(np.arange(1,13)+0.125, pm_2016 - np.mean(mat, axis=0), width=0.25, color='r', label='2016 mon anom')
plt.xlim([0,13])
plt.xticks(np.arange(1,13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
plt.legend()
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")

plt.tight_layout()
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/PM_annual_cycle.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()



