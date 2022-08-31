import pandas as pd
import numpy as np
import xarray as xr

df = pd.read_csv('../../../hyperdrought_data/series/nino34.long.anom.data.txt', sep='\s+', skiprows=1, skipfooter=7, engine='python', header=None)
iniyear = str(int(df.iloc[0,0]))
endyear = str(int(df.iloc[-1,0]))
datetime = pd.date_range(start=iniyear+'-01-01', end=endyear+'-12-31', freq='MS')
da = xr.DataArray(np.ravel(df.iloc[:,1:]), coords=[datetime], dims=['time'])
da[da==-99.99] = np.nan