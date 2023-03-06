import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import statsmodels.api as sm

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

df = pd.read_excel("../../../hyperdrought_data/forest/nbr_shp_19112022.dbf.xlsx")
n,m  = df.shape

data = df.iloc[:,4:]
columns = list(data.columns)

f = lambda x : True if x == '01' else False
summer = [f(col[-2:]) for col in columns]
winter = [not sum for sum in summer]  

nmax_summer = 2022-1986+1
nmax_winter = 2021-1986+1

no_nans_summer = np.zeros((n,))
no_nans_winter = np.zeros((n,))

full_data = []

for i in range(n):
    
    no_nans_summer[i] = np.count_nonzero(~np.isnan(data.iloc[i,summer].values))
    no_nans_winter[i] = np.count_nonzero(~np.isnan(data.iloc[i,winter].values))

    if nmax_summer - no_nans_summer[i] == 0:
        full_data = full_data + [i]

meta = df.iloc[full_data,:4]
data = data.iloc[full_data, summer]

df_pr = pd.read_excel("../../../hyperdrought_data/forest/precip_data.xlsx")
df_tmin = pd.read_excel("../../../hyperdrought_data/forest/tmin_data.xlsx")
df_tmax = pd.read_excel("../../../hyperdrought_data/forest/tmax_data.xlsx")

data_pr = df_pr.iloc[:,4:].iloc[full_data, summer]

data_tmin = df_tmin.iloc[:,4:].iloc[full_data, summer]
data_tmax = df_tmax.iloc[:,4:].iloc[full_data, summer]

nn, mm = data.shape

mat_pr = np.zeros((nn,))
mat_tmin = np.zeros((nn,))
mat_tmax = np.zeros((nn,))

for i in range(nn):
    mat_pr[i] = np.corrcoef(data.iloc[i,:-1].values, data_pr.iloc[i,:-1].values)[0,1]
    mat_tmin[i] = np.corrcoef(data.iloc[i,:-1].values, data_tmin.iloc[i,:-1].values)[0,1]
    mat_tmax[i] = np.corrcoef(data.iloc[i,:-1].values, data_tmax.iloc[i,:-1].values)[0,1]

ds = xr.open_dataset("../../../hyperdrought_data/CR2MET/CR2MET_pr_v2.5_mon_1960_2021_005deg.nc")
da = ds['pr']
winter_months = da.sel(time=slice('1985-07-01', '2022-10-31')).where(da.time.dt.month.isin([7,8,9,10]), drop=True)
winter_series = winter_months.rolling(time=4, center=True).sum()[2::4]
mat_pr_prev = np.zeros((nn,))

pr_prev = np.zeros((nn, mm))

for i in range(meta.shape[0]):
    row = meta.iloc[i,:]
    ws = winter_series.sel(lat=row['lat'], lon=row['long'], method='nearest')
    pr_prev[i, : ] = ws.values
    mat_pr_prev[i] = np.corrcoef(data.iloc[i,:-1].values, ws[:-1].values)[0,1]

fig, axs = plt.subplots(2,2, figsize=(12,7))
plt.sca(axs[0,0])
plt.hist(mat_pr, bins = np.arange(-1, 1.05, 0.05), color='b')
plt.xlim([-1,1])
plt.xlabel('Correlacion nbr verano con precip verano')
plt.ylabel('Frecuencia')
plt.sca(axs[0,1])
plt.hist(mat_pr_prev, bins = np.arange(-1, 1.05, 0.05), color='b')
plt.xlim([-1,1])
plt.xlabel('Correlacion nbr verano con precip invierno previo')
plt.ylabel('Frecuencia')
plt.sca(axs[1,0])
plt.hist(mat_tmin, bins = np.arange(-1, 1.05, 0.05), color='b')
plt.xlim([-1,1])
plt.xlabel('Correlacion nbr verano con tmin verano')
plt.ylabel('Frecuencia')
plt.sca(axs[1,1])
plt.hist(mat_tmax, bins = np.arange(-1, 1.05, 0.05), color='b')
plt.xlim([-1,1])
plt.xlabel('Correlacion nbr verano con tmax verano')
plt.ylabel('Frecuencia')
plt.tight_layout()
# plt.savefig('../../../hyperdrought_data/forest/corr_summer_hist.png', dpi=300)
plt.show()

def h(x):
    x = x.iloc[:,:-1]
    y = x.copy()
    for i in range(x.shape[0]):
        y.iloc[i, :] = x.iloc[i, :].values - np.mean(x.iloc[i, :].values)
    return y.values.T

def g(x):
    x = x.iloc[:,:-1]
    y = x.copy()
    for i in range(x.shape[0]):
        y.iloc[i, :] = x.iloc[i, :].values/np.mean(x.iloc[i, :].values)*100-100
    return y.values.T

def f(x):
    x = x[:,:-1]
    y = x.copy()
    for i in range(x.shape[0]):
        y[i, :] = x[i, :]/np.mean(x[i, :])*100-100
    return y.T

fig, axs = plt.subplots(5,1, figsize=(10,9))
plt.sca(axs[0])
plt.plot(np.arange(1986, 2022), h(data), color='grey', lw=0.5)
plt.plot(np.arange(1986, 2022), np.mean(h(data), axis=1), color='b', lw=1.5)
plt.xlim([1985, 2022])
plt.axhline(0, color='red', lw=1.0)
plt.ylabel('NBR ver anom')
plt.xlim([1986, 2021])

plt.sca(axs[1])
plt.plot(np.arange(1986, 2022), g(data_pr), color='grey', lw=0.5)
plt.plot(np.arange(1986, 2022), np.mean(g(data_pr), axis=1), color='b', lw=1.5)
plt.xlim([1986, 2021])
plt.ylabel('Pr ver anom (%)')
plt.axhline(0, color='red', lw=1.0)

plt.sca(axs[2])
plt.plot(np.arange(1986, 2022), f(pr_prev), color='grey', lw=0.5)
plt.plot(np.arange(1986, 2022), np.mean(f(pr_prev), axis=1), color='b', lw=1.5)
plt.xlim([1986, 2021])
plt.ylabel('Pr inv prev. anom (%)')
plt.axhline(0, color='red', lw=1.0)

plt.sca(axs[3])
plt.plot(np.arange(1986, 2022), -1*h(data_tmin), color='grey', lw=0.5)
plt.plot(np.arange(1986, 2022), -1*np.mean(h(data_tmin), axis=1), color='b', lw=1.5)
plt.xlim([1986, 2021])
plt.ylabel('Tmin ver anom inv (-1xºC)')
plt.axhline(0, color='red', lw=1.0)


plt.sca(axs[4])
plt.plot(np.arange(1986, 2022), -1*h(data_tmax), color='grey', lw=0.5)
plt.plot(np.arange(1986, 2022), -1*np.mean(h(data_tmax), axis=1), color='b', lw=1.5)
plt.xlim([1986, 2021])
plt.axhline(0, color='red', lw=1.0)
plt.tight_layout()
plt.ylabel('Tmax ver anom (-1xºC)')
plt.savefig('../../../hyperdrought_data/forest/corr_summer_series.png', dpi=300)
plt.show()

def mullin(x1, x2, y):
    x1 = x1[:,:-1]
    x2 = x2.iloc[:,:-1]
    y = y.iloc[:,:-1]
    n = x1.shape[0]
    mat = np.zeros((n,))
    for i in range(n):

        x1a = np.reshape(x1[i,:], (-1,1))
        x2a = np.reshape(x2.iloc[i,:].values, (-1,1))
        ya = np.reshape(y.iloc[i,:].values, (-1,1))

        X = np.append(x1a, x2a, axis=1)
        X = sm.add_constant(X)
        results = sm.OLS(ya, X).fit()
        mat[i] = results.rsquared
    return mat

mat = mullin(pr_prev, data_tmax, data)









