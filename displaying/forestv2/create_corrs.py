import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

df_pr = pd.read_excel("../../../hyperdrought_data/forestv2/pr_data.xlsx")
df_pr_yr = pd.read_excel("../../../hyperdrought_data/forestv2/pr_annual_data.xlsx")
df_pr_prevseas = pd.read_excel("../../../hyperdrought_data/forestv2/pr_prevseas_data.xlsx")
df_nbr = pd.read_excel("../../../hyperdrought_data/forestv2/nbr_20012023_metadata.xlsx")
df_ndvi = pd.read_excel("../../../hyperdrought_data/forestv2/ndvi_20012023_metadata.xlsx")
df_tmin = pd.read_excel("../../../hyperdrought_data/forestv2/tmin_data.xlsx")
df_tmax = pd.read_excel("../../../hyperdrought_data/forestv2/tmax_data.xlsx")

n, m = df_nbr.shape
columns = list(df_nbr.columns)

df_blueprint = df_nbr.iloc[:, :4]

df_nbr_pr = df_blueprint.copy()
df_nbr_pr_yr = df_blueprint.copy()
df_nbr_tmin = df_blueprint.copy()
df_nbr_tmax = df_blueprint.copy()
df_nbr_pr_prevseas = df_blueprint.copy()

df_ndvi_pr = df_blueprint.copy()
df_ndvi_pr_yr = df_blueprint.copy()
df_ndvi_tmin = df_blueprint.copy()
df_ndvi_tmax = df_blueprint.copy()
df_ndvi_pr_prevseas = df_blueprint.copy()

for season in ['01', '02']:

    alist = []
    for j in range(4, m):
        label = columns[j]
        year = label[1:5]
        seas = label[-2:]
        if seas == season and year != '2022':
            alist = alist + [j]

    n_nbr = np.zeros((n,))
    n_ndvi = np.zeros((n,))
    
    r_pr_nbr = np.zeros((n,))*np.nan
    r_pr_yr_nbr = np.zeros((n,))*np.nan
    r_pr_prevseas_nbr = np.zeros((n,))*np.nan
    r_tmax_nbr = np.zeros((n,))*np.nan
    r_tmin_nbr = np.zeros((n,))*np.nan
    
    p_pr_nbr = np.zeros((n,))*np.nan
    p_pr_yr_nbr = np.zeros((n,))*np.nan
    p_pr_prevseas_nbr = np.zeros((n,))*np.nan
    p_tmax_nbr = np.zeros((n,))*np.nan
    p_tmin_nbr = np.zeros((n,))*np.nan
    
    r_pr_ndvi = np.zeros((n,))*np.nan
    r_pr_yr_ndvi = np.zeros((n,))*np.nan
    r_pr_prevseas_ndvi = np.zeros((n,))*np.nan
    r_tmax_ndvi = np.zeros((n,))*np.nan
    r_tmin_ndvi = np.zeros((n,))*np.nan
    
    p_pr_ndvi = np.zeros((n,))*np.nan
    p_pr_yr_ndvi = np.zeros((n,))*np.nan
    p_pr_prevseas_ndvi = np.zeros((n,))*np.nan
    p_tmax_ndvi = np.zeros((n,))*np.nan
    p_tmin_ndvi = np.zeros((n,))*np.nan
    
    # from 1992 to 2021
    for i in range(n):
        pr = df_pr.iloc[i,alist].values[6:]
        pr_yr = df_pr_yr.iloc[i,alist].values[6:]
        pr_prevseas = df_pr_prevseas.iloc[i,alist].values[6:]
        tmin = df_tmin.iloc[i,alist].values[6:]
        tmax = df_tmax.iloc[i,alist].values[6:]
        nbr = df_nbr.iloc[i,alist].values[6:]
        ndvi = df_ndvi.iloc[i,alist].values[6:]
        
        mask_nbr = ~np.isnan(nbr)
        mask_ndvi = ~np.isnan(ndvi)
        
        nbr_nbr = nbr[mask_nbr]
        pr_nbr = pr[mask_nbr]
        pr_yr_nbr = pr_yr[mask_nbr]
        pr_prevseas_nbr = pr_prevseas[mask_nbr]
        tmin_nbr = tmin[mask_nbr]
        tmax_nbr = tmax[mask_nbr]
        
        ndvi_ndvi = ndvi[mask_ndvi]
        pr_ndvi = pr[mask_ndvi]
        pr_yr_ndvi = pr_yr[mask_ndvi]
        pr_prevseas_ndvi = pr_prevseas[mask_ndvi]
        tmin_ndvi = tmin[mask_ndvi]
        tmax_ndvi = tmax[mask_ndvi]
        
        n_nbr[i] = nbr_nbr.size
        n_ndvi[i] = ndvi_ndvi.size
        
        if n_nbr[i] >= 10:
        
            r_pr_nbr[i], p_pr_nbr[i] = pearsonr(nbr_nbr, pr_nbr)
            r_pr_yr_nbr[i], p_pr_yr_nbr[i] = pearsonr(nbr_nbr, pr_yr_nbr)
            r_pr_prevseas_nbr[i], p_pr_prevseas_nbr[i] = pearsonr(nbr_nbr, pr_prevseas_nbr)
            r_tmin_nbr[i], p_tmin_nbr[i] = pearsonr(nbr_nbr, tmin_nbr)
            r_tmax_nbr[i], p_tmax_nbr[i] = pearsonr(nbr_nbr, tmax_nbr)
            
        if n_ndvi[i] >= 10:
        
            r_pr_ndvi[i], p_pr_ndvi[i] = pearsonr(ndvi_ndvi, pr_ndvi)
            r_pr_yr_ndvi[i], p_pr_yr_ndvi[i] = pearsonr(ndvi_ndvi, pr_yr_ndvi)
            r_pr_prevseas_ndvi[i], p_pr_prevseas_ndvi[i] = pearsonr(ndvi_ndvi, pr_prevseas_ndvi)
            r_tmin_ndvi[i], p_tmin_ndvi[i] = pearsonr(ndvi_ndvi, tmin_ndvi)
            r_tmax_ndvi[i], p_tmax_ndvi[i] = pearsonr(ndvi_ndvi, tmax_ndvi)
        
    df_nbr_pr['r_'+season] = r_pr_nbr
    df_nbr_pr_yr['r_'+season] = r_pr_yr_nbr
    df_nbr_pr_prevseas['r_'+season] = r_pr_prevseas_nbr
    df_nbr_tmin['r_'+season] = r_tmin_nbr
    df_nbr_tmax['r_'+season] = r_tmax_nbr
    
    df_nbr_pr['p_'+season] = p_pr_nbr
    df_nbr_pr_yr['p_'+season] = p_pr_yr_nbr
    df_nbr_pr_prevseas['p_'+season] = p_pr_prevseas_nbr
    df_nbr_tmin['p_'+season] = p_tmin_nbr
    df_nbr_tmax['p_'+season] = p_tmax_nbr
    
    df_nbr_pr['n_'+season] = n_nbr
    df_nbr_pr_yr['n_'+season] = n_nbr
    df_nbr_pr_prevseas['n_'+season] = n_nbr
    df_nbr_tmin['n_'+season] = n_nbr
    df_nbr_tmax['n_'+season] = n_nbr
    
    
    df_ndvi_pr['r_'+season] = r_pr_ndvi
    df_ndvi_pr_yr['r_'+season] = r_pr_yr_ndvi
    df_ndvi_pr_prevseas['r_'+season] = r_pr_prevseas_ndvi
    df_ndvi_tmin['r_'+season] = r_tmin_ndvi
    df_ndvi_tmax['r_'+season] = r_tmax_ndvi
    
    df_ndvi_pr['p_'+season] = p_pr_ndvi
    df_ndvi_pr_yr['p_'+season] = p_pr_yr_ndvi
    df_ndvi_pr_prevseas['p_'+season] = p_pr_prevseas_ndvi
    df_ndvi_tmin['p_'+season] = p_tmin_ndvi
    df_ndvi_tmax['p_'+season] = p_tmax_ndvi
    
    df_ndvi_pr['n_'+season] = n_ndvi
    df_ndvi_pr_yr['n_'+season] = n_ndvi
    df_ndvi_pr_prevseas['n_'+season] = n_ndvi
    df_ndvi_tmin['n_'+season] = n_ndvi
    df_ndvi_tmax['n_'+season] = n_ndvi


df_nbr_pr.to_csv('../../../hyperdrought_data/forestv2/corr_nbr_pr.csv')
df_nbr_pr_yr.to_csv('../../../hyperdrought_data/forestv2/corr_nbr_pr_yr.csv')
df_nbr_pr_prevseas.to_csv('../../../hyperdrought_data/forestv2/corr_nbr_pr_prevseas.csv')
df_nbr_tmin.to_csv('../../../hyperdrought_data/forestv2/corr_nbr_tmin.csv')
df_nbr_tmax.to_csv('../../../hyperdrought_data/forestv2/corr_nbr_tmax.csv')
 
df_ndvi_pr.to_csv('../../../hyperdrought_data/forestv2/corr_ndvi_pr.csv')
df_ndvi_pr_yr.to_csv('../../../hyperdrought_data/forestv2/corr_ndvi_pr_yr.csv')
df_ndvi_pr_prevseas.to_csv('../../../hyperdrought_data/forestv2/corr_ndvi_pr_prevseas.csv')
df_ndvi_tmin.to_csv('../../../hyperdrought_data/forestv2/corr_ndvi_tmin.csv')
df_ndvi_tmax.to_csv('../../../hyperdrought_data/forestv2/corr_ndvi_tmax.csv')
    
    
        
        
        
        
        
    
    
    
    