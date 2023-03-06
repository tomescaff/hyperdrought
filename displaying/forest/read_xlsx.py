import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

fig, axs = plt.subplots(1,2, figsize=(12,7))
plt.sca(axs[0])
plt.hist(nmax_summer - no_nans_summer, bins = np.arange(-0.5, 75, 0.5))
# cuantos faltan para estar completo
plt.xlabel('Datos faltantes')
plt.ylabel('Frecuencia')
plt.sca(axs[1])
plt.hist(nmax_winter - no_nans_winter, bins = np.arange(-0.5, 75, 0.5))
plt.xlabel('Datos faltantes')
plt.ylabel('Frecuencia')
plt.show()