import pandas as pd

df = pd.read_csv('../../../hyperdrought_data/series/ersst5.nino.mth.91-20.ascii.txt', sep='\s+', parse_dates={'time': ['YR', 'MON']})
df = df.rename({'ANOM.3':'NINO3.4a'}, axis='columns')
df = df.set_index('time')
da = df['NINO3.4a'].to_xarray()
da = da.sel(time=slice('1950', '2021'))
da = da.where(da.time.dt.month.isin([6,7,8]), drop=True)
da = da.resample(time='1YS').sum('time')
