import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

relpath = '../../../hyperdrought_data/output/'
obs_mle_1880 = pd.read_csv(join(relpath, '2019ee_metrics_of_attr_QN_MLE_1880_2019_fit_1930_2021.csv'), index_col=0)
# lens2_mle_1880 = pd.read_csv(join(relpath, 'metrics_of_attr_LENS2_MLE_1880_2017_normal_fit_1930_2021.csv'), index_col=0)

models = [ 
            obs_mle_1880,
            obs_mle_1880,
            obs_mle_1880, 
         ]
model_names = [
            'QN',
            'LENS1',
            'LENS2']

center_rr = np.array([x.loc['rr c-a', 'raw'] for x in models])
lower_rr = np.array([x.loc['rr c-a', '95ci lower'] for x in models])
upper_rr = np.array([x.loc['rr c-a', '95ci upper'] for x in models])
width_rr = upper_rr - lower_rr

center_ac = np.array([x.loc['tau ac', 'raw'] for x in models])
lower_ac = np.array([x.loc['tau ac', '95ci lower'] for x in models])
upper_ac = np.array([x.loc['tau ac', '95ci upper'] for x in models])
width_ac = upper_ac - lower_ac

center_cf = np.array([x.loc['tau cf', 'raw'] for x in models])
lower_cf = np.array([x.loc['tau cf', '95ci lower'] for x in models])
upper_cf = np.array([x.loc['tau cf', '95ci upper'] for x in models])
width_cf = upper_cf - lower_cf

fig, axs = plt.subplots(3,1, sharex = True, figsize=(12,5))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
y_pos = np.arange(len(model_names))

ax=axs[0]
plt.sca(ax)
barlist = ax.barh(y_pos, width=width_cf, left=lower_cf, height=0.4, align='center')
colors=['blue']#, 'blue', 'blue', 'red', 'red', 'red', 'green', 'green']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center_cf, y_pos, s=200, marker='|', color='k', zorder=4)
plt.yticks(y_pos, model_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Tau 1880')
ax.set_xscale('log')
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
#plt.xlim([0.001, 100000])
plt.axvline(1, color='k', lw=0.4, ls='--')

ax = axs[1]
plt.sca(ax)
barlist = ax.barh(y_pos, width=width_ac, left=lower_ac, height=0.4, align='center')
colors=['blue']#, 'blue', 'blue', 'red', 'red', 'red', 'green', 'green']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center_ac, y_pos, s=200, marker='|', color='k', zorder=4)
plt.yticks(y_pos, model_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Tau 2017')
ax.set_xscale('log')
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
#plt.xlim([0.001, 100000])
plt.axvline(1, color='k', lw=0.4, ls='--')

ax = axs[2]
plt.sca(ax)
barlist = ax.barh(y_pos, width=width_rr, left=lower_rr, height=0.4, align='center')
colors=['blue']#, 'blue', 'blue', 'red', 'red', 'red', 'green', 'green']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center_rr, y_pos, s=200, marker='|', color='k', zorder=4)
plt.yticks(y_pos, model_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Probability ratio')
ax.set_xscale('log')
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
plt.xlim([0.1, 10000000000])
plt.axvline(1, color='k', lw=0.4, ls='--')


plt.tight_layout()

#plt.savefig('../../../megafires_data/png/QN_sysnthesis_plot_observed_metrics_rr.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()




