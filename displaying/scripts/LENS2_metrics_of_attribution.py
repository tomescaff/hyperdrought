import sys
import numpy as np
import pandas as pd
from scipy.stats import gamma

from sklearn.utils import resample as bootstrap

sys.path.append('../../processing')

import processing.series as se
import processing.lens as lens 

index = ['tau cf', 'tau ac', 'tau fu', 'rr c-a', 'far c-a', 'rr a-f', 'far a-f']
columns = ['raw', '95ci lower', '95ci upper', '1percentile']
df = pd.DataFrame(columns=columns, index=index)

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()
qn_1866_2021 = qn.sel(time=slice('1866', '2021'))
qn_mean_1866_2021 = qn_1866_2021.mean('time')
qn_anom = qn.sel(time='2019') - qn.sel(time=slice('1991', '2020')).mean('time')

# get LENS2 time series ensamble
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()
lens2_mean_1866_2021 = lens2.sel(time=slice('1866', '2021')).mean('time')
lens2_anom = lens2 - lens2_mean_1866_2021
lens2_fixed_1866_2021 = lens2_anom + qn_mean_1866_2021

# get past period
da_jan_1851_1880 = lens2_fixed_1866_2021.sel(time=slice('1851','1880'))
np_jan_all_runs_1851_1880 = np.ravel(da_jan_1851_1880.values)

# get 1991-2020 period
da_jan_1991_2020 = lens2_fixed_1866_2021.sel(time=slice('1991', '2020'))
np_jan_all_runs_1991_2020 = np.ravel(da_jan_1991_2020.values)

# get 2071-2100 period
da_jan_2071_2100 = lens2_fixed_1866_2021.sel(time=slice('2071', '2100'))
np_jan_all_runs_2071_2100 = np.ravel(da_jan_2071_2100.values)

# fitting distributions
g2fit_1851_1880 = gamma.fit(np_jan_all_runs_1851_1880, floc=0)
g2fit_1991_2020 = gamma.fit(np_jan_all_runs_1991_2020, floc=0)
g2fit_2071_2100 = gamma.fit(np_jan_all_runs_2071_2100, floc=0)

# get anom 
ev = qn_anom + np.mean(np_jan_all_runs_1991_2020)
tau_1991_2020 = 1/gamma.cdf(ev, *g2fit_1991_2020)
tau_1851_1880 = 1/gamma.cdf(ev, *g2fit_1851_1880)
tau_2071_2100 = 1/gamma.cdf(ev, *g2fit_2071_2100)

rr_ca = tau_1851_1880/tau_1991_2020
rr_af = tau_1991_2020/tau_2071_2100

far_ca = (tau_1851_1880-tau_1991_2020)/tau_1851_1880
far_af = (tau_1991_2020-tau_2071_2100)/tau_1991_2020

df.loc['tau cf', 'raw'] = tau_1851_1880
df.loc['tau ac', 'raw'] = tau_1991_2020
df.loc['tau fu', 'raw'] = tau_2071_2100
df.loc['rr c-a', 'raw'] = rr_ca
df.loc['far c-a', 'raw'] = far_ca
df.loc['rr a-f', 'raw'] = rr_af
df.loc['far a-f', 'raw'] = far_af

# bootstraping LENS2
nboot = 100000

bspreds_tau_1851_1880 = np.zeros((nboot,))
bspreds_tau_1991_2020 = np.zeros((nboot,))
bspreds_tau_2071_2100 = np.zeros((nboot,))
bspreds_rr_ca = np.zeros((nboot,))
bspreds_far_ca = np.zeros((nboot,))
bspreds_rr_af = np.zeros((nboot,))
bspreds_far_af = np.zeros((nboot,))

for i in range(nboot):
    z_1851_1880_i = bootstrap(np_jan_all_runs_1851_1880)
    z_1991_2020_i = bootstrap(np_jan_all_runs_1991_2020)
    z_2071_2100_i = bootstrap(np_jan_all_runs_2071_2100)

    g2fit_1851_1880_i = gamma.fit(z_1851_1880_i, floc=0, scale=1)
    g2fit_1991_2020_i = gamma.fit(z_1991_2020_i, floc=0, scale=1)
    g2fit_2071_2100_i = gamma.fit(z_2071_2100_i, floc=0, scale=1)

    tau_1851_1880_i = 1/gamma.cdf(ev, *g2fit_1851_1880_i)
    tau_1991_2020_i = 1/gamma.cdf(ev, *g2fit_1991_2020_i)
    tau_2071_2100_i = 1/gamma.cdf(ev, *g2fit_2071_2100_i)

    rr_ca_i = tau_1851_1880_i/tau_1991_2020_i
    far_ca_i = (tau_1851_1880_i-tau_1991_2020_i)/tau_1851_1880_i

    rr_af_i = tau_1991_2020_i/tau_2071_2100_i
    far_af_i = (tau_1991_2020_i-tau_2071_2100_i)/tau_1991_2020_i

    bspreds_tau_1851_1880[i] = tau_1851_1880_i
    bspreds_tau_1991_2020[i] = tau_1991_2020_i
    bspreds_tau_2071_2100[i] = tau_2071_2100_i
    bspreds_rr_ca[i] = rr_ca_i
    bspreds_far_ca[i] = far_ca_i
    bspreds_rr_af[i] = rr_af_i
    bspreds_far_af[i] = far_af_i

mapping = [('95ci lower', 0.025), ('95ci upper', 0.975), ('1percentile', 0.01)]

for col, thr in mapping:
    df.loc['tau cf', col] = np.quantile(bspreds_tau_1851_1880, [thr], axis = 0)
    df.loc['tau ac', col] = np.quantile(bspreds_tau_1991_2020, [thr], axis = 0)
    df.loc['tau fu', col] = np.quantile(bspreds_tau_2071_2100, [thr], axis = 0)
    df.loc['rr c-a', col] = np.quantile(bspreds_rr_ca, [thr], axis = 0)
    df.loc['far c-a', col] = np.quantile(bspreds_far_ca, [thr], axis = 0)
    df.loc['rr a-f', col] = np.quantile(bspreds_rr_af, [thr], axis = 0)
    df.loc['far a-f', col] = np.quantile(bspreds_far_af, [thr], axis = 0)

df = df.applymap(lambda x: round(float(x),2))






