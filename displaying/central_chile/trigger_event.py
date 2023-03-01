import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

basedir = '../../../hyperdrought_data/DMC/'

relpaths = \
[
"DMC_acc_annual_precip_calama_90p.csv",
"DMC_acc_annual_precip_la_serena_100p.csv",
"DMC_acc_annual_precip_tobalaba_80p.csv",
"DMC_acc_annual_precip_quinta_normal_90p.csv",
"DMC_acc_annual_precip_pudahuel_80p.csv",
"DMC_acc_annual_precip_juan_fernandez_90p.csv",
"DMC_acc_annual_precip_santo_domingo_80p.csv",
"DMC_acc_annual_precip_curico_90p.csv",
"DMC_acc_annual_precip_chillan_90p.csv",
"DMC_acc_annual_precip_concepcion_90p.csv",
"DMC_acc_annual_precip_temuco_90p.csv",
"DMC_acc_annual_precip_valdivia_90p.csv",
"DMC_acc_annual_precip_osorno_90p.csv",
"DMC_acc_annual_precip_pto_montt.csv",
"DMC_acc_annual_precip_coyhaique_90p.csv",
"DMC_acc_annual_precip_balmaceda_80p.csv",
"DMC_acc_annual_precip_chile_chico_80p.csv",
"DMC_acc_annual_precip_cochrane_80p.csv",
"DMC_acc_annual_precip_punta_arenas_80p.csv",
]

names = \
[
"Calama",
"La Serena",
"Tobalaba",
"Quinta Normal",
"Pudahuel",
"Juan Fernandez",
"Santo Domingo",
"Curicó",
"Chillán",
"Concepción",
"Temuco",
"Valdivia",
"Osorno",
"Puerto Montt",
"Coyhaique",
"Balmaceda",
"Chile Chico",
"Cochrane",
"Punta Arenas",
]


filepaths = [basedir + relpath for relpath in relpaths]

def read_csv(filepath):

    df = pd.read_csv(filepath, parse_dates=['agno'] )
    year = df['agno'].values
    prec = df[' valor'].values
    
    da = xr.DataArray(prec, coords=[year], dims=['time'])

    return da

dataarrays = [read_csv(filepath) for filepath in filepaths]

clims = np.array([da.sel(time=slice('1991','2020')).mean('time').values for da in dataarrays])

val2019 = np.array([float(da.sel(time='2019').values) for da in dataarrays])/clims*100-100

medians = np.array([np.quantile(da.values, 0.5) for da in dataarrays])/clims*100-100
mins = np.array([np.quantile(da.values, 0.0) for da in dataarrays])/clims*100-100
maxs = np.array([np.quantile(da.values, 1.0) for da in dataarrays])/clims*100-100
means = np.array([np.mean(da.values) for da in dataarrays])/clims*100-100

# TODO PONER 2021 cuando se pueda

y = np.arange(len(names))

fig = plt.figure(figsize=(8,8))

plt.barh(y, maxs-mins, 0.75, left = mins, color='grey')
plt.barh(y, 0-val2019, 0.75, left = val2019, color='brown')
plt.yticks(y, names)
plt.gca().invert_yaxis()
plt.xlim([-100, 0])
plt.tight_layout()
plt.show()

