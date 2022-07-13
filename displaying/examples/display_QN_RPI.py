import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Arial'
import sys

sys.path.append('../../processing')

import processing.series as se

# get Quinta Normal time series
ds = se.get_QN_RPI()
qn = ds['QN']
rpiv1 = ds['RPIv1']
rpiv2 = ds['RPIv2']

# create figure
fig = plt.figure(figsize=(12,6))

# plot the data
plt.plot(qn.time.values, qn.values, lw = 1,  color='blue', label='QN')
plt.plot(rpiv1.time.values, rpiv1.values, lw = 1,  color='grey', label='RPIv1')
plt.plot(rpiv2.time.values, rpiv2.values, lw = 1,  color='r', label='RPIv2')

# set grid
plt.grid(lw=0.2, ls='--', color='grey')

# set legend
plt.legend()

# set title and labels
plt.xlabel('Time (years)')
plt.ylabel('Precipitation (s.u.)')
plt.savefig('../../../hyperdrought_data/png/display_QN_RPI.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()