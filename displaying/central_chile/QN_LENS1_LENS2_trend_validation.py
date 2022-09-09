import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import linregress
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se
import processing.lens as lens

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record().sel(time=slice('1866', '2021'))

# get LENS1 time series
lens1 = lens.get_LENS1_annual_precip_NOAA_QNEW()

# get lens2 data
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()

# common period
qn_norm = qn.sel(time=slice('1930','2021'))/qn.sel(time=slice('1980','2020')).mean('time')*100
lens1_norm = lens1.sel(time=slice('1930','2021'))/lens1.sel(time=slice('1980','2020')).mean('time')*100
lens2_norm = lens2.sel(time=slice('1930','2021'))/lens2.sel(time=slice('1980','2020')).mean('time')*100

# qn trend CI
def obs_trend(xr_array, ini_year, end_year):
    xr_array = xr_array.sel(time=slice(ini_year,end_year))
    x = xr_array.time.dt.year.values
    y = xr_array.values
    n = xr_array.size
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_hat = x*slope + intercept
    x_bar = np.mean(x)
    SE = np.sqrt(np.sum((y-y_hat)**2)/(n-2))/np.sqrt(np.sum((x-x_bar)**2))
    alpha = 0.05
    dof = n-2
    p_star = 1-alpha/2
    ME_trend = t.ppf(p_star, dof)*SE
    return slope, ME_trend

def mod_trends(series, ini_year, end_year):
    series = series.sel(time=slice(ini_year, end_year))
    nruns, ntime = series.shape
    series_trends = np.zeros((nruns,))
    for k in range(nruns):
        y = series[k,:].values
        x = series[k,:].time.dt.year.values
        series_trends[k] = linregress(x,y).slope
    return series_trends

qn_1930_2021, ME_1930_2021 = obs_trend(qn_norm, '1930', '2021')
lens1_trends_1930_2021 = mod_trends(lens1, '1930', '2021')
lens2_trends_1930_2021 = mod_trends(lens2, '1930', '2021')

qn_1960_2016, ME_1960_2016 = obs_trend(qn_norm, '1960', '2016')
lens1_trends_1960_2016 = mod_trends(lens1, '1960', '2016')
lens2_trends_1960_2016 = mod_trends(lens2, '1960', '2016')

qn_1980_2021, ME_1980_2021 = obs_trend(qn_norm, '1980', '2021')
lens1_trends_1980_2021 = mod_trends(lens1, '1980', '2021')
lens2_trends_1980_2021 = mod_trends(lens2, '1980', '2021')

# plot
fig, axs = plt.subplots(1,3, figsize=(15,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

plt.sca(axs[0])
plt.errorbar(x=0.2, y=qn_1930_2021*10, yerr=ME_1930_2021*10, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.boxplot(lens1_trends_1930_2021*10, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(lens2_trends_1930_2021*10, notch=False, meanline=True,showmeans=True, positions=[1.4], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
# plt.ylim([-350,100])
plt.xlim([-0.1,1.7])
plt.ylabel('Norm. annual precip at QN - 1930-2021 linear trend (%/dec)')
plt.xticks([0.2,0.8, 1.4], ["DMC (95% CI)","LENS1", "LENS2"], rotation=0)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")
plt.title('Trend 1930-2021')

plt.sca(axs[1])
plt.errorbar(x=0.2, y=qn_1960_2016*10, yerr=ME_1960_2016*10, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.boxplot(lens1_trends_1960_2016*10, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(lens2_trends_1960_2016*10, notch=False, meanline=True,showmeans=True, positions=[1.4], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
# plt.ylim([-350,100])
plt.xlim([-0.1,1.7])
plt.ylabel('Norm. annual precip at QN - 1960-2016 linear trend (%/dec)')
plt.xticks([0.2,0.8, 1.4], ["DMC (95% CI)","LENS1", "LENS2"], rotation=0)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")
plt.title('Trend 1960-2016')

plt.sca(axs[2])
plt.errorbar(x=0.2, y=qn_1980_2021*10, yerr=ME_1980_2021*10, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.boxplot(lens1_trends_1980_2021*10, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(lens2_trends_1980_2021*10, notch=False, meanline=True,showmeans=True, positions=[1.4], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
# plt.ylim([-350,100])
plt.xlim([-0.1,1.7])
plt.ylabel('Norm. annual precip at QN - 1980-2021 linear trend (%/dec)')
plt.xticks([0.2,0.8, 1.4], ["DMC (95% CI)","LENS1", "LENS2"], rotation=0)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().tick_params(direction="in")
plt.title('Trend 1980-2021')

plt.tight_layout()
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/QN_LENS1_LENS2_trend_validation.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()