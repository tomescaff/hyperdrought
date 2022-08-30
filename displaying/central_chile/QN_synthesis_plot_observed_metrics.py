import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/output/'

moa_30y = pd.read_csv(join(currentdir, relpath, '2019ee_metrics_of_attr_QN_30y.csv'), index_col=0)
moa_50y = pd.read_csv(join(currentdir, relpath, '2019ee_metrics_of_attr_QN_50y.csv'), index_col=0)
moa_mle_1880 = pd.read_csv(join(currentdir, relpath, '2019ee_metrics_of_attr_QN_MLE_1880_2019.csv'), index_col=0)

models = [  moa_30y, 
            moa_50y, 
            moa_mle_1880 ]
model_names = (  
            '30yr - Gamma (1880 vs. 2019)', 
            '50yr - Gamma (1880 vs. 2019)', 
            'MLE - Gamma (1880 vs. 2019)')

center = np.array([x.loc['rr c-a', 'raw'] for x in models])
lower = np.array([x.loc['rr c-a', '95ci lower'] for x in models])
upper = np.array([x.loc['rr c-a', '95ci upper'] for x in models])
width = upper - lower

# rr
fig = plt.figure(figsize=(12,2))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
y_pos = np.arange(len(model_names))
ax = plt.gca()
barlist = ax.barh(y_pos, width=width, left=lower, height=0.4, align='center')
colors=['blue', 'blue', 'blue']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color='fuchsia', zorder=4)
plt.yticks(y_pos, model_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Probability ratio')
ax.set_xscale('log')
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
plt.xlim([0.001, 100000])
plt.tight_layout()
plt.axvline(1, color='k', lw=0.4, ls='--')
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/2019ee_QN_sysnthesis_plot_observed_metrics_rr.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

center = np.array([x.loc['tau ac', 'raw'] for x in models])
lower = np.array([x.loc['tau ac', '95ci lower'] for x in models])
upper = np.array([x.loc['tau ac', '95ci upper'] for x in models])
width = upper - lower

# return periods
fig = plt.figure(figsize=(12,2))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
y_pos = np.arange(len(model_names))
ax = plt.gca()
barlist = ax.barh(y_pos, width=width, left=lower, height=0.4, align='center')
colors=['blue', 'blue', 'blue']
for bar, color in zip(barlist, colors):
    bar.set_color(color)
plt.scatter(center, y_pos, s=200, marker='|', color='fuchsia', zorder=4)
plt.yticks(y_pos, model_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Return period (yr)')
ax.set_xscale('log')
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
plt.xlim([0.001, 100000])
plt.tight_layout()
plt.axvline(1, color='k', lw=0.4, ls='--')
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/2019ee_QN_sysnthesis_plot_observed_metrics_tau.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()