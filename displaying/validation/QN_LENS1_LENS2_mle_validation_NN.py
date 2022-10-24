import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from os.path import join

relpath = '../../../hyperdrought_data/output/'

filename = 'MLE_precip_QN_GMST_10000_validation.nc'
qn = xr.open_dataset(join(relpath, filename))

filename = 'MLE_2019ee_LENS1_GMST_100_validation_QN_NN.nc'
lens1 = xr.open_dataset(join(relpath, filename))

filename = 'MLE_2019ee_LENS2_GMST_100_validation_QN_NN.nc'
lens2 = xr.open_dataset(join(relpath, filename))

model_names = ['observations', 'lens1', 'lens2']
models = [qn, lens1, lens2]

fig, axs = plt.subplots(3,1,figsize=(12,6))
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
colors=['blue', 'blue', 'blue']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color='k', zorder=4)
plt.yticks(y_pos, model_names)
# plt.xlim([28,30])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(varname)
ax.set_xlabel('mu0')
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
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
colors=['blue', 'blue', 'blue']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color='k', zorder=4)
plt.yticks(y_pos, model_names)
# plt.xlim([0.0,2.0])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(varname)
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
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
colors=['blue', 'blue', 'blue']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color='k', zorder=4)
plt.yticks(y_pos, model_names)
#plt.xlim([0.5, 2.5])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(varname)
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)

plt.tight_layout()
plt.savefig('../../../hyperdrought_data/png/QN_LENS1_LENS2_mle_validation_NN.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()


