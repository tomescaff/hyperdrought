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

xx, yy, dd = ut.get_nearest_90p_contour(nino34a.values, precip.values, -2, 2, 0, 220)
r, p = pearsonr(nino34a.values, precip.values)

# scatter full period
fig = plt.figure(figsize=(8,8))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.contourf(xx,yy,dd,cmap='Greys',levels=[0,4.61],alpha=0.2)
plt.contour(xx,yy,dd,colors=['grey'],levels=[9.21], linewidths=[0.5], linestyles='--')
plt.scatter(nino34a.values, precip.values, color='k', alpha=0.2, label=f'Full period r = {r:0.2f}*')
plt.scatter(nino34a.sel(time=slice('2010', '2021')).values, precip.sel(time=slice('2010', '2021')).values, color='r', alpha=1)
plt.scatter(nino34a.sel(time='2019').values, precip.sel(time='2019').values, color='fuchsia', alpha=1)
plt.scatter(nino34a.mean('time'), precip.mean('time'), s=50, color='k', alpha=0.8, marker='+', label='1870-2021 mean')
plt.scatter(nino34a.sel(time=slice('2010', '2021')).mean('time'), precip.sel(time=slice('2010', '2021')).mean('time'), s=50, color='r', alpha=1, marker='+', label='2010-2021 mean')
plt.axhline(100, color='grey', ls='--', lw=1.0)
plt.axvline(0, color='grey', ls='--', lw=1.0)
plt.axhline(80, color='blue', ls='--', lw=1.2)
plt.xlim([-1.8, 1.8])
plt.ylim([10, 220])
plt.yticks(np.arange(20, 240, 20))
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('MJJAS Niño3.4 index [ºC]')
plt.ylabel('Quinta Normal norm precipitation (ref. 1981-2010) [%]')
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/enso_precip_scatter.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()


xx, yy, dd_past = ut.get_nearest_90p_contour(nino34a.sel(time=slice('1870', '1920')), precip.sel(time=slice('1870', '1920')), -2, 2, 0, 220)
xx, yy, dd_pres = ut.get_nearest_90p_contour(nino34a.sel(time=slice('1971', '2021')), precip.sel(time=slice('1971', '2021')), -2, 2, 0, 220)

r_past, p_past = pearsonr(nino34a.sel(time=slice('1870', '1920')).values, precip.sel(time=slice('1870', '1920')).values)
r_pres, p_pres = pearsonr(nino34a.sel(time=slice('1971', '2021')).values, precip.sel(time=slice('1971', '2021')).values)

# scatter first 50 vs last 50 years
fig = plt.figure(figsize=(8,8))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.contour(xx,yy,dd_past,colors=['b'],levels=[4.61], linewidths=[0.5], linestyles='--')
plt.contour(xx,yy,dd_pres,colors=['r'],levels=[4.61], linewidths=[0.5], linestyles='--')
plt.scatter(nino34a.sel(time=slice('1870', '1920')).values, precip.sel(time=slice('1870', '1920')).values, color='b', alpha=0.2, label=f'1870-1920 r = {r_past:0.2f}*')
plt.scatter(nino34a.sel(time=slice('1971', '2021')).values, precip.sel(time=slice('1971', '2021')).values, color='r', alpha=0.2, label=f'1971-2021 r = {r_pres:0.2f}*')
plt.scatter(nino34a.sel(time=slice('1870', '1920')).mean('time'), precip.sel(time=slice('1870', '1920')).mean('time'), s=50, color='b', alpha=0.8, marker='+', label='1870-1920 mean')
plt.scatter(nino34a.sel(time=slice('1971', '2021')).mean('time'), precip.sel(time=slice('1971', '2021')).mean('time'), s=50, color='r', alpha=0.8, marker='+', label='1971-2021 mean')
plt.axhline(100, color='grey', ls='--', lw=1.0)
plt.axvline(0, color='grey', ls='--', lw=1.0)
plt.axhline(80, color='blue', ls='--', lw=1.2)
plt.xlim([-1.8, 1.8])
plt.ylim([10, 220])
plt.yticks(np.arange(20, 240, 20))
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('MJJAS Niño3.4 index [ºC]')
plt.ylabel('Quinta Normal norm precipitation (ref. 1981-2010) [%]')
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/enso_precip_50yr_comparison_scatter.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

def confidence_interval(r, n):
    alpha = 0.05
    p_star = 1-alpha/2
    z_star = norm.ppf(p_star)
    z_r = 1/2*np.log((1+r)/(1-r))
    z_l = z_r - z_star*np.sqrt(1/(n-3))
    z_u = z_r + z_star*np.sqrt(1/(n-3))
    r_l = (np.exp(2*z_l)-1)/(np.exp(2*z_l)+1)
    r_u = (np.exp(2*z_u)-1)/(np.exp(2*z_u)+1)
    return r_l, r_u

r_l, r_u = confidence_interval(r, precip.size)
r_past_l, r_past_u = confidence_interval(r_past, precip.sel(time=slice('1870', '1920')).size)
r_pres_l, r_pres_u = confidence_interval(r_pres, precip.sel(time=slice('1971', '2021')).size)

fig = plt.figure(figsize=(6,6))
plt.errorbar(x=0, y=r, yerr=[[r-r_l], [r_u-r]], lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.errorbar(x=1, y=r_past, yerr=[[r_past-r_past_l], [r_past_u-r_past]], lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.errorbar(x=2, y=r_pres, yerr=[[r_pres-r_pres_l], [r_pres_u-r_pres]], lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.xlim([-1,3])
plt.ylim([0,1])
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.ylabel('Correlation (95% CI)')
plt.yticks(np.arange(0,1.1,0.1))
plt.xticks([0,1,2], ['Full period', '1870-1920', '1971-2021'])
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/enso_precip_50yr_comparison_95p_CI.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()



