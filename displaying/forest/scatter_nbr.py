import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

df_pr = pd.read_excel("../../../hyperdrought_data/forest/precip_data.xlsx")
df_nbr = pd.read_excel("../../../hyperdrought_data/forest/nbr_shp_19112022.dbf.xlsx")
df_tmin = pd.read_excel("../../../hyperdrought_data/forest/tmin_data.xlsx")
df_tmax = pd.read_excel("../../../hyperdrought_data/forest/tmax_data.xlsx")

data_pr = df_pr.iloc[:,4:]
data_nbr = df_nbr.iloc[:,4:]
data_tmin = df_tmin.iloc[:,4:]
data_tmax = df_tmax.iloc[:,4:]

columns = list(data_nbr.columns)

f = lambda x : True if x == '01' else False
summer = [f(col[-2:]) for col in columns]
winter = [not sum for sum in summer]   


def create_series(data_nbr, data_pr, data_tmin, data_tmax, mask):

    seas_nbr = data_nbr.iloc[:,mask].copy()
    seas_pr = data_pr.iloc[:,mask].copy() 
    seas_tmin = data_tmin.iloc[:,mask].copy() 
    seas_tmax = data_tmax.iloc[:,mask].copy()

    # creating pr anomalies
    n, m = seas_pr.shape

    for i in range(n):
        x = seas_pr.iloc[i,:].values
        seas_pr.iloc[i,:] = seas_pr.iloc[i,:].values/np.nanmean(x)*100 -100

    for i in range(n):
        x = seas_tmin.iloc[i,:].values
        seas_tmin.iloc[i,:] = seas_tmin.iloc[i,:].values-np.nanmean(x)

    for i in range(n):
        x = seas_tmax.iloc[i,:].values
        seas_tmax.iloc[i,:] = seas_tmax.iloc[i,:].values-np.nanmean(x)
 
    rav_nbr = np.ravel(seas_nbr.values)
    rav_pr = np.ravel(seas_pr.values)
    rav_tmin = np.ravel(seas_tmin.values)
    rav_tmax = np.ravel(seas_tmax.values)

    mask_nbr = ~np.isnan(rav_nbr)
    mask_pr = ~np.isnan(rav_pr)
    mask_tmin = ~np.isnan(rav_tmin)
    mask_tmax = ~np.isnan(rav_tmax)

    x_pr = rav_nbr[mask_nbr & mask_pr]
    y_pr = rav_pr[mask_nbr & mask_pr]

    x_tmin = rav_nbr[mask_nbr & mask_tmin]
    y_tmin = rav_tmin[mask_nbr & mask_tmin]

    x_tmax = rav_nbr[mask_nbr & mask_tmax]
    y_tmax = rav_tmax[mask_nbr & mask_tmax]

    return x_pr, y_pr, x_tmin, y_tmin, x_tmax, y_tmax

sx_pr, sy_pr, sx_tmin, sy_tmin, sx_tmax, sy_tmax = create_series(data_nbr, data_pr, data_tmin, data_tmax, summer)
wx_pr, wy_pr, wx_tmin, wy_tmin, wx_tmax, wy_tmax = create_series(data_nbr, data_pr, data_tmin, data_tmax, winter)

fig, axs = plt.subplots(2, 3, figsize=(13,7))
plt.sca(axs[0, 0])
plt.scatter(sx_pr, sy_pr, 10, color='b', alpha=0.125)
plt.xlabel('NBR verano')
plt.ylabel('Anom. precip verano (%)')
plt.xlim([-0.3, 1])
plt.sca(axs[0, 1])
plt.scatter(sx_tmin, sy_tmin, 10, color='b', alpha=0.125)
plt.xlabel('NBR verano')
plt.ylabel('Anom. tmin verano (ºC)')
plt.xlim([-0.3, 1])
plt.ylim([-1.7, 1.7])
plt.sca(axs[0, 2])
plt.scatter(sx_tmax, sy_tmax, 10, color='b', alpha=0.125)
plt.xlabel('NBR verano')
plt.ylabel('Anom. tmax verano (ºC)')
plt.xlim([-0.3, 1])
plt.ylim([-2.5, 2.5])
plt.sca(axs[1, 0])
plt.scatter(wx_pr, wy_pr, 10, color='b', alpha=0.125)
plt.xlabel('NBR invierno')
plt.ylabel('Anom. precip invierno (%)')
plt.xlim([-0.3, 1])
plt.sca(axs[1, 1])
plt.scatter(wx_tmin, wy_tmin, 10, color='b', alpha=0.125)
plt.xlabel('NBR invierno')
plt.ylabel('Anom. tmin invierno (ºC)')
plt.xlim([-0.3, 1])
plt.ylim([-1.7, 1.7])
plt.sca(axs[1, 2])
plt.scatter(wx_tmax, wy_tmax, 10, color='b', alpha=0.125)
plt.xlabel('NBR invierno')
plt.ylabel('Anom. tmax invierno (ºC)')
plt.xlim([-0.3, 1])
plt.ylim([-2.5, 2.5])
plt.tight_layout()
plt.savefig('../../../hyperdrought_data/forest/scatter_nbr.png', dpi=300)
plt.show()
