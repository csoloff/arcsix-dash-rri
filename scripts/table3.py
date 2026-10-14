#%% Requirements
from scripts.config import output_path
import pandas as pd
import numpy as np

# Read
d = pd.read_csv(output_path+'/merged_data.csv', parse_dates=['Time_Start','Time_Stop','Time_Mid'])

# Data filtering
d = d[d.TF == 0].reset_index(drop=True)
d = d[d.InletFlag_LARGE.isin([0])].reset_index(drop=True)
d = d[d.lwc<0.001].reset_index(drop=True)

#%% Table 3

def calc_stats(df):
    df.loc[i,'N'] = len(di)
    for key in avg_keys:
        mean = di[key].mean()
        median = di[key].median()
        std = di[key].std()
        df.loc[i, key] = f"{median:.2f}/{mean:.2f} ({std:.3f})"

o = pd.DataFrame() # output

o['Research flight']=list(np.linspace(1,10,10).astype(int))+list(np.linspace(12,19,8).astype(int))

dates = d.Time_Start.dt.date.unique()
o['Date'] = dates
o['Date_str'] = pd.to_datetime(dates).strftime('%-m-%d')

avg_keys = ['Org_phi', 'H2SO4_phi','NH4HSO4_phi','NH42SO4_phi','NH4NO3_phi','RI']
for i in range(len(o)):
    date = o.loc[i,'Date']
    di = d[d.Time_Start.dt.date == date]
    calc_stats(o)


t = pd.DataFrame() # total
dsp = d[d.season=='spr']
dsm = d[d.season=='sum']
d_sea = [dsp, dsm, d]

for i in range(len(d_sea)):
    di = d_sea[i]
    calc_stats(t)

o.to_excel(output_path+'/table3_flights_stats.xlsx')
t.to_excel(output_path+'/table3_season_stats.xlsx')