import sys
import numpy as np
import pandas as pd
from scipy.stats import gamma

from sklearn.utils import resample as bootstrap

sys.path.append('../../processing')

import processing.series as se

index = ['tau cf', 'tau ac', 'rr c-a', 'far c-a', 'delta c-a']
columns = ['raw', '95ci lower', '95ci upper', '1percentile']
df = pd.DataFrame(columns=columns, index=index)

# get Quinta Normal time series
qn = se.get_CO_JFM_precip()

# get past period
da_cf = qn.sel(time=slice('1954','1983'))
np_cf = np.ravel(da_cf.values)

# get 1991-2020 period
da_ac = qn.sel(time=slice('1992','2021'))
np_ac = np.ravel(da_ac.values)

# fitting distributions
fit_cf = gamma.fit(np_cf, floc=0, scale=1)
fit_ac = gamma.fit(np_ac, floc=0, scale=1)

# get ev value 
ev = qn.sel(time='2016')

# get return periods
tau_cf = 1/gamma.cdf(ev, *fit_cf)
tau_ac = 1/gamma.cdf(ev, *fit_ac)

# get rr and far
rr_ca = tau_cf/tau_ac
far_ca = (tau_cf-tau_ac)/tau_cf

# get delta
delta = ev - gamma.ppf(1/tau_ac, *fit_cf)

df.loc['tau cf', 'raw'] = tau_cf
df.loc['tau ac', 'raw'] = tau_ac
df.loc['rr c-a', 'raw'] = rr_ca
df.loc['far c-a', 'raw'] = far_ca
df.loc['delta c-a', 'raw'] = delta

# bootstraping 
nboot = 100000

bspreds_tau_cf = np.zeros((nboot,))
bspreds_tau_ac = np.zeros((nboot,))

bspreds_rr_ca = np.zeros((nboot,))
bspreds_far_ca = np.zeros((nboot,))
bspreds_delta = np.zeros((nboot,))

for i in range(nboot):
    z_cf_i = bootstrap(np_cf)
    z_ac_i = bootstrap(np_ac)

    fit_cf_i = gamma.fit(z_cf_i, floc=0, scale=1)
    fit_ac_i = gamma.fit(z_ac_i, floc=0, scale=1)

    tau_cf_i = 1/gamma.cdf(ev, *fit_cf_i)
    tau_ac_i = 1/gamma.cdf(ev, *fit_ac_i)

    rr_ca_i = tau_cf_i/tau_ac_i
    far_ca_i = (tau_cf_i-tau_ac_i)/tau_cf_i
    delta_i = ev - gamma.ppf(1/tau_ac_i, *fit_cf_i)

    bspreds_tau_cf[i] = tau_cf_i
    bspreds_tau_ac[i] = tau_ac_i
    bspreds_rr_ca[i] = rr_ca_i
    bspreds_far_ca[i] = far_ca_i
    bspreds_delta[i] = delta_i

mapping = [('95ci lower', 0.025), ('95ci upper', 0.975), ('1percentile', 0.01)]

for col, thr in mapping:
    df.loc['tau cf', col] = np.quantile(bspreds_tau_cf, [thr], axis = 0)
    df.loc['tau ac', col] = np.quantile(bspreds_tau_ac, [thr], axis = 0)
    df.loc['rr c-a', col] = np.quantile(bspreds_rr_ca, [thr], axis = 0)
    df.loc['far c-a', col] = np.quantile(bspreds_far_ca, [thr], axis = 0)
    df.loc['delta c-a', col] = np.quantile(bspreds_delta, [thr], axis = 0)

df = df.applymap(lambda x: round(float(x),2))