import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import gamma
from scipy.stats import exponweib
from scipy.stats import invgamma
from scipy.stats import beta
from scipy.stats import lognorm
from scipy.stats import weibull_min

sys.path.append('../../processing')

import processing.series as se

# get Quinta Normal time series
ds = se.get_QN_RPI()
qn = ds['QN']
qn = qn.dropna('time')

# create figure
fig, axs = plt.subplots(3, 1, figsize=(10,7.5), constrained_layout=True)

# plt params
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8

# compute histogram
xmin, xmax, step = -3.75,3.75, 0.125
density = True
hist_all, bins = np.histogram(qn.values, bins=np.arange(xmin, xmax+step, step), density=density)
hist_first, bins = np.histogram(qn.values[:70], bins=np.arange(xmin, xmax+step, step), density=density)
hist_last, bins = np.histogram(qn.values[-70:], bins=np.arange(xmin, xmax+step, step), density=density)


# plot the histogram
width = 0.9 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
axs[0].bar(center, hist_all, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'QN Full Period')
axs[1].bar(center, hist_first, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'QN 70 first years')
axs[2].bar(center, hist_last, align='center', width=width, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75, label = 'QN 70 last years')

# plot pdfs
x = np.linspace(xmin, xmax, 100)
for dist, col, lab in zip((gamma, weibull_min, invgamma, lognorm), ('b', 'k', 'r', 'g'), ('gamma', 'weibull_min', 'invgamma', 'lognorm')):
    axs[0].plot(x, dist.pdf(x, *dist.fit(qn.values)), lw=1.0, color=col, label=lab)
    axs[1].plot(x, dist.pdf(x, *dist.fit(qn.values[:70])), lw=1.0, color=col, label=lab)
    axs[2].plot(x, dist.pdf(x, *dist.fit(qn.values[-70:])), lw=1.0, color=col, label=lab)

for ax in axs:
    plt.sca(ax)
    plt.grid(lw=0.2, ls='--', color='grey') # set grid
    plt.legend() # set legend
    plt.xticks(np.arange(-3.5,3.5+0.5,0.5))
    plt.yticks(np.arange(0,0.9+0.2,0.2))
    plt.xlim([xmin, xmax]) # set xlim
    plt.ylim([0, 0.9]) # set ylim
    plt.xlabel('Annual Precip (s.u.)') # set labels
    plt.ylabel('PDF')

print(list(zip(qn.time.dt.year[qn.values.argsort()][:4].values, qn[qn.values.argsort()][:4].values)))
print(list(zip(qn.time.dt.year[qn.values.argsort()][-4:].values, qn[qn.values.argsort()][-4:].values)))

plt.savefig('../../../hyperdrought_data/png/QN_distribution.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()