import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/output/'

filename = 'PM_MLEv2_precip_PM_GMST_10_validation.nc'
qn = xr.open_dataset(join(currentdir, relpath, filename))

filename = 'PM_MLEv2_precip_LENS1_GMST_10_validation_NN.nc'
lens1 = xr.open_dataset(join(currentdir, relpath, filename))

filename = 'PM_MLEv2_precip_LENS2_GMST_10_validation_NN.nc'
lens2 = xr.open_dataset(join(currentdir, relpath, filename))

model_names = ['OBS', 'CESM1-LENS', 'CESM2-LENS', 'EC_Earth3', 'CMIP6']
models = [qn, lens1, lens2, lens2, lens2]

fig, axs = plt.subplots(3,1,figsize=(8,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

# sigma0
varname = 'sigma0'
center = [np.quantile(m[varname].values*m['eta'].values, 0.5, axis=0) for m in models]
lower  = [np.quantile(m[varname].values*m['eta'].values, 0.025, axis=0) for m in models]
upper  = [np.quantile(m[varname].values*m['eta'].values, 0.975, axis=0) for m in models]
width  = np.array(upper) - np.array(lower)


plt.sca(axs[0])
y_pos = np.arange(len(model_names))
ax = plt.gca()
barlist = ax.barh(y_pos, width=width, left=lower, height=0.4, align='center')
colors=['blue', 'blue', 'blue', 'none', 'none']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color=['k', 'k', 'k', 'none', 'none'], zorder=4)
plt.yticks(y_pos, model_names)
plt.xlim([0.0,1.5])
ax.set_axisbelow(True)
plt.grid( lw=0.4, ls='--', color='grey', zorder=-4)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('mu0')
ax.spines.right.set_visible(False)
ax.tick_params(direction="in")
ax.spines.top.set_visible(False)

# eta
varname = 'eta'
center = [np.quantile(m[varname].values, 0.5, axis=0) for m in models]
lower  = [np.quantile(m[varname].values, 0.025, axis=0) for m in models]
upper  = [np.quantile(m[varname].values, 0.975, axis=0) for m in models]
width  = np.array(upper) - np.array(lower)

plt.sca(axs[1])
y_pos = np.arange(len(model_names))
ax = plt.gca()
barlist = ax.barh(y_pos, width=width, left=lower, height=0.4, align='center')
colors=['blue', 'blue', 'blue', 'none', 'none']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color=['k', 'k', 'k', 'none', 'none'], zorder=4)
plt.yticks(y_pos, model_names)
plt.xlim([0,20])
ax.set_axisbelow(True)
plt.grid( lw=0.4, ls='--', color='grey', zorder=-4)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(varname)
ax.spines.right.set_visible(False)
ax.tick_params(direction="in")
ax.spines.top.set_visible(False)

# alpha
varname = 'alpha'
center = [np.quantile(m[varname].values, 0.5, axis=0) for m in models]
lower  = [np.quantile(m[varname].values, 0.025, axis=0) for m in models]
upper  = [np.quantile(m[varname].values, 0.975, axis=0) for m in models]
width  = np.array(upper) - np.array(lower)

plt.sca(axs[2])
y_pos = np.arange(len(model_names))
ax = plt.gca()
barlist = ax.barh(y_pos, width=width, left=lower, height=0.4, align='center')
colors=['blue', 'blue', 'blue', 'none', 'none']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color=['k', 'k', 'k', 'none', 'none'], zorder=4)
plt.yticks(y_pos, model_names)
plt.xlim([-0.8, 0.8])
ax.set_axisbelow(True)
ax.invert_yaxis()  # labels read top-to-bottom
plt.grid( lw=0.4, ls='--', color='grey', zorder=-4)
ax.set_xlabel(varname)
ax.spines.right.set_visible(False)
ax.tick_params(direction="in")
ax.spines.top.set_visible(False)

plt.tight_layout()
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/PM_all_models_mle_validation_2016ee.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()