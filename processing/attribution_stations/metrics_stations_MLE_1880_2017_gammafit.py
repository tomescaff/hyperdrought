import sys
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import gamma
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.math as pmath

# raw values

smf = gmst.get_gmst_annual_5year_smooth()
filepath = '../../../hyperdrought_data/CR2_explorer/cr2_pr_stns_data_before_1990_24deg_40degS.nc'
stnsf = xr.open_dataset(join(currentdir, filepath))

N = stnsf.stn.size

for j in range(N):

    index = ['tau cf', 'tau ac', 'rr c-a', 'far c-a', 'delta c-a']
    columns = ['raw', '95ci lower', '95ci upper', '1percentile']
    df = pd.DataFrame(columns=columns, index=index)

    ac_year = '2017'
    cf_year = '1880'

    prf = stnsf.data[:, j]
    name = str(stnsf.name[j].values)

    prev_year = int(xr.where(np.isnan(prf), 1,np.nan).dropna('time').time.dt.year[-1].values)
    pr = prf.sel(time=slice(str(prev_year+1), '2021'))
    sm = smf.sel(time=slice(str(prev_year+1), '2021'))
    
    sm_no2019 = sm.where(sm.time.dt.year != 2019, drop=True)
    pr_n02019 = pr.where(pr.time.dt.year != 2019, drop=True)

    xopt = pmath.mle_gamma_2d_fast(pr_n02019.values, sm_no2019.values, [70, 4, -0.5])

    sigma0, eta, alpha = xopt
    sigma = sigma0*np.exp(alpha*smf)

    sig_MLE_ac = sigma.sel(time = ac_year)
    sig_MLE_cf = sigma.sel(time = cf_year)
    eta_MLE = eta

        # get ev value 
    ev = pr.sel(time='2019')

    # get return periods
    tau_cf = 1/gamma.cdf(ev, eta, 0, sig_MLE_cf)
    tau_ac = 1/gamma.cdf(ev, eta, 0, sig_MLE_ac)

    # get rr and far
    rr_ca = tau_cf/tau_ac
    far_ca = (tau_cf-tau_ac)/tau_cf

    # get delta
    ev0 = gamma.ppf(1/tau_ac, eta, 0, sig_MLE_cf)
    delta = 100*(ev - ev0)/ev0

    df.loc['tau cf', 'raw'] = tau_cf
    df.loc['tau ac', 'raw'] = tau_ac
    df.loc['rr c-a', 'raw'] = rr_ca
    df.loc['far c-a', 'raw'] = far_ca
    df.loc['delta c-a', 'raw'] = delta

    # bootstrap MLE
    nboot = 10
    filepath = f'../../../hyperdrought_data/output/MLE_2019ee_GMST_nboot_{nboot}_gamma_stn_{j:02}.nc'
    bspreds = xr.open_dataset(filepath)
    bspreds_sigma0 = bspreds.sigma0.values
    bspreds_eta = bspreds.eta.values
    bspreds_alpha = bspreds.alpha.values

    Tac = smf.sel(time = ac_year).values
    Tcf = smf.sel(time = cf_year).values

    sig_ac_dist = bspreds_sigma0*np.exp(bspreds_alpha*Tac)
    sig_cf_dist = bspreds_sigma0*np.exp(bspreds_alpha*Tcf)
    eta_dist = bspreds_eta

    bspreds_tau_cf = np.zeros((nboot,))
    bspreds_tau_ac = np.zeros((nboot,))
    bspreds_rr_ca = np.zeros((nboot,))
    bspreds_far_ca = np.zeros((nboot,))
    bspreds_delta = np.zeros((nboot,))

    for i in range(nboot):
    
        tau_cf_i = 1/gamma.cdf(ev, eta_dist[i], 0, sig_cf_dist[i])
        tau_ac_i = 1/gamma.cdf(ev, eta_dist[i], 0, sig_ac_dist[i])

        rr_ca_i = tau_cf_i/tau_ac_i
        far_ca_i = (tau_cf_i-tau_ac_i)/tau_cf_i
        ev0_i = gamma.ppf(1/tau_ac_i, eta_dist[i], 0, sig_cf_dist[i])
        delta_i = 100*(ev - ev0_i)/ev0_i

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
    fileout = f'../../../hyperdrought_data/output/metrics_MLE_{cf_year}_{ac_year}_gammafit_stn_{j:02}.csv'
    df.to_csv(join(currentdir, fileout))