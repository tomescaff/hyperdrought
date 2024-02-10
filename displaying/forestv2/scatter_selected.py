import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



df_pr = pd.read_excel("../../../hyperdrought_data/forestv2/pr_data.xlsx")
df_pr_yr = pd.read_excel("../../../hyperdrought_data/forestv2/pr_annual_data.xlsx")
df_nbr = pd.read_excel("../../../hyperdrought_data/forestv2/nbr_20012023_metadata.xlsx")
df_ndvi = pd.read_excel("../../../hyperdrought_data/forestv2/ndvi_20012023_metadata.xlsx")
df_tmin = pd.read_excel("../../../hyperdrought_data/forestv2/tmin_data.xlsx")
df_tmax = pd.read_excel("../../../hyperdrought_data/forestv2/tmax_data.xlsx")

df_index = pd.read_csv('summer_row_full_data.csv', index_col = 0)

x = int(np.random.randint(0, df_index.size))
index = df_index.iloc[x,0]
columns = list(df_nbr.columns)
season = '01'
lat, lon = df_nbr.iloc[index, [2,3]]
alist = []
for j in range(4, len(columns)):
    label = columns[j]
    year = label[1:5]
    seas = label[-2:]
    if seas == season and year != '2022':
        alist = alist + [j]
        
# from 1992 to 2021
pr = df_pr.iloc[index,alist].values[6:]
pr_yr = df_pr_yr.iloc[index,alist].values[6:]
tmin = df_tmin.iloc[index,alist].values[6:]
tmax = df_tmax.iloc[index,alist].values[6:]
nbr = df_nbr.iloc[index,alist].values[6:]
ndvi = df_ndvi.iloc[index,alist].values[6:]

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
fig, axs = plt.subplots(2,4, figsize=(12,6))
axs_ = np.ravel(axs)
names_list = [(x, y) for x in ["nbr", "ndvi"] for y in ["pr", "pr_yr", "tmin", "tmax"] ]
data_list = [(x, y) for x in [nbr, ndvi] for y in [pr, pr_yr, tmin, tmax]]

for data, names, ax in zip(data_list, names_list, axs_):
    xlabel, ylabel = names
    x, y = data
    ax.scatter(x, y, s=20, c='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend([f'r = {np.corrcoef(x,y)[0,1]:02f}'], prop={'size': 6})
    
plt.suptitle(f'Scatter plots for season {season} at {lat:01f},{lon:01f}')
plt.tight_layout()
plt.show()
    


