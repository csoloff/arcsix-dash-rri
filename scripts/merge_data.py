#%% Requirements

import glob
import pandas as pd
from datetime import datetime as dt
import numpy as np
from scripts.config import data_path, output_path
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import root_scalar
from sklearn.linear_model import LinearRegression

def simple_read(path, na_rep = np.nan, na_val = -9999):
    '''
    Reads .ict files to a Pandas DataFrame
    :param path: path to the .ict data
    :return: Pandas DataFrame with .ict data
    '''
    with open(path) as f:
        # find the value in the file which tells you how many lines to skip to get to the table
        first_line = f.readline()
        header_line = int(first_line[0:-2].split(",")[0])-1
    data = pd.read_csv(path, sep=',', skiprows=header_line)

    # finds the location in the path containing the date
    acc = 0
    boo = False
    for letter in path:
        if letter == '2':
            boo = True
        elif boo and letter == '0':
            acc -= 1
            break
        acc += 1
        
    # creates datetime object with the date the data was collected
    day = dt(int(path[acc:acc+4]), int(path[acc+4:acc+6]), int(path[acc+6:acc+8])) 
    
    for column in data.keys():
        if 'Time' in column:
            # converts seconds after midnight columns to datetime
            data[column] = day + pd.to_timedelta(data[column], unit='seconds')
    data.columns = data.columns.str.replace(' ', '')
    return data.replace(na_val, na_rep) # Converts -9999 values to na_rep value (default np.nan)

def read_instr(instr, subset = None, na_rep = np.nan, path=data_path+'/*', na_val = -9999):
    paths = sorted(glob.glob(path+instr+'*'))
    d_list = []
    for i in range(0, len(paths)):
        d_list.append(simple_read(paths[i], na_rep = na_rep, na_val = na_val))
    d = pd.concat(d_list).reset_index(drop=True)
    if subset:
        d = d.dropna(subset = subset, how='all').reset_index(drop=True)
    return d

def merge_data(d1, d2, n_jobs=-1, unq=False, agg_func='mean'):

    def row_mean(d1_row, d2, vars):
        # Filter d2 based on the time range in d1_row
        sub = d2[(d2['Time_Start'] >= d1_row['Time_Start']) & (d2['Time_Stop'] <= d1_row['Time_Stop'])].agg(agg_func)
        # Return the index and the computed values
        return sub[vars]
    
    def row_mean_single_time(d1_row, d2, vars, time_key):
        # Filter d2 based on the time range in d1_row
        sub = d2[(d2[time_key] >= d1_row['Time_Start']) & (d2[time_key] < d1_row['Time_Stop'])].agg(agg_func)
        # Return the index and the computed values
        return sub[vars]

    def row_unq_single_time(d1_row, d2, vars, time_key):
        # Filter d2 based on the time range in d1_row
        sub = d2[(d2[time_key] >= d1_row['Time_Start']) & (d2[time_key] < d1_row['Time_Stop'])].nunique(dropna=False)
        # Return the index and the computed values
        return sub[vars]


    # Get the list of variables to merge (excluding 'Time' related columns)
    vars = [var for var in d2.keys() if 'Time' not in var]
    
    if 'Time_Start' and 'Time_Stop' in d2.keys():
        # Use joblib to parallelize the processing of each row in d1
        m = Parallel(n_jobs=n_jobs)(
            delayed(row_mean)(d1_row, d2, vars) for _, d1_row in tqdm(d1.iterrows(), total = len(d1))
        )
    else:
        time_key = [var for var in d2.keys() if 'Time' in var][0]
        m = Parallel(n_jobs=n_jobs)(
            delayed(row_mean_single_time)(d1_row, d2, vars, time_key) for _, d1_row in tqdm(d1.iterrows(), total = len(d1))
        )

    m = pd.concat(m, axis=1).T
    out = pd.merge(d1, m, left_index=True, right_index=True)

    if unq: # adds a count for unique values
        m2 = Parallel(n_jobs=n_jobs)(
            delayed(row_unq_single_time)(d1_row, d2, vars, time_key) for _, d1_row in tqdm(d1.iterrows(), total = len(d1))
        )
        m2 = pd.concat(m2, axis=1).T.add_suffix('_unq')
        out = pd.merge(out, m2, left_index=True, right_index=True)
    return out

#%% Read instrument data

DASH = read_instr('DASH').dropna(subset=['Dp']).reset_index(drop=True)
AMS = read_instr('AMS')
OPT = read_instr('OPT')
MetNav = read_instr('MetNav')
micro = read_instr('MICRO')
sp2 = read_instr('SP2')
UHSAS = read_instr('UHSAS')
inlet = read_instr('InletFlag', na_rep=-1) # -1 represents data marked as having bad inlet flow
FCDP = read_instr('FCDP', na_val=-999)
ATOF = read_instr('A-ATOFMS')

#%% Merging datasets

d = merge_data(DASH, AMS)
d = merge_data(d, OPT)
d = merge_data(d, MetNav)
d = merge_data(d, micro)
d = merge_data(d, sp2)
d = merge_data(d, UHSAS)
d = merge_data(d, inlet, unq=True)
d.loc[d['InletFlag_LARGE_unq']>1, 'InletFlag_LARGE'] = -1 # marking merged inlets as unstable flow
def qutl9(d_func, q=.9):
    return d_func.quantile(q)
def qutl10(d_func, q=.1):
    return d_func.quantile(q)
d = merge_data(d, FCDP, agg_func=qutl9)

# Merge ATOF data (different method because lower resolution the DASH)
ATOF_keys = ['NumFrac_AgedSmoke',
       'NumFrac_Dust', 'NumFrac_SeaSalt', 'NumFrac_SulfuricAcid',
       'NumFrac_Soot', 'NumFrac_Other', 'NumFrac_AgedSmoke_Error',
       'NumFrac_Dust_Error', 'NumFrac_SeaSalt_Error',
       'NumFrac_SulfuricAcid_Error', 'NumFrac_Soot_Error',
       'NumFrac_Other_Error']
for i in range(len(ATOF)):
    mask = (d['Time_Start'] >= ATOF.loc[i,'Time_Start']) & (d['Time_Stop'] <= ATOF.loc[i,'Time_Stop'])
    d.loc[mask, ATOF_keys] = ATOF.loc[i,ATOF_keys].values
    d.loc[mask, 'ATOF_samp'] = i

#%% Calculate kappa

def g_k_eq(k, D_d, gf, RH):
    A = 4*0.072*0.0180153/(8.3144598*(273.15+17)*1000)
    return (gf**3 - 1) / (gf**3 - (1 - k)) - RH / np.exp(A/(D_d*gf))

def get_k(D_d, gf, RH):
    return root_scalar(g_k_eq, args=(D_d, gf, RH), bracket=[0, 2], method='brentq').root

d['k_dash'] = np.nan
for i in tqdm(range(len(d))):
    if np.isfinite(d.loc[i, 'GF']) & np.isfinite(d.loc[i, 'RH']):
        row = d.loc[i]
        if row.GF <= 1:
            d.loc[i, 'k_dash'] = 0
        else:
            d.loc[i, 'k_dash'] = get_k(row.Dp*1E-9, row.GF, row.RH/100)

#%% Categorize transit flights

# Convert the dates to datetime objects for comparison
target_dates = ['5/24/2024', '6/17/2024', '7/22/2024', '08/16/2024']
target_dates = pd.to_datetime(target_dates).date

# Set 'TF' to 1 where 'Time_Mid' is in the target dates
d['TF'] = d['Time_Mid'].dt.date.isin(target_dates).astype(int)

#%% ZSR mixing rule

for k_i in ['Org_AMS_STP', 'SO4_AMS_STP', 'NO3_AMS_STP', 'NH4_AMS_STP', 'Chl_AMS_STP', 'BC_mass', 'mz44_AMS_STP']:
    d.loc[d[k_i]<0, k_i] = 0

class species:
    def __init__(self, name, molar_mass, density=None):
        self.name = name
        self.mass = molar_mass
        self.density = density

NH4NO3 = species('NH4NO3', 80.043, 1.72)
H2SO4 = species('H2SO4', 98.08, 1.83)
NH4HSO4 = species('NH4HSO4', 115.11, 1.78)
NH42SO4 = species('NH42SO4', 132.14, 1.77)

all_species = [NH4NO3, H2SO4, NH4HSO4, NH42SO4]

SO4 = species('SO4_AMS_STP', 96.06)
NO3 = species('NO3_AMS_STP', 62.0049)
NH4 = species('NH4_AMS_STP', 18.04)
Chl = species('Chl_AMS_STP', 35.45)

all_AMS = [SO4, NO3, NH4, Chl]

for i in range(0,len(all_AMS)):
    s = all_AMS[i]
    d[s.name[0:3]+'_m'] = d[s.name]*1E-6 / s.mass

d[NH4NO3.name] = (d['NO3_m']) * NH4NO3.mass
d[H2SO4.name] = np.maximum(0, d['SO4_m']-d['NH4_m']+d['NO3_m']) * H2SO4.mass
d[NH4HSO4.name] = np.minimum(2*d['SO4_m'] - d['NH4_m'] + d['NO3_m'], d['NH4_m'] - d['NO3_m']) * NH4HSO4.mass
d.loc[d[NH4HSO4.name] < 0, NH4HSO4.name] = 0
d[NH42SO4.name] = np.maximum(0, d['NH4_m'] - d['NO3_m'] - d['SO4_m']) * NH42SO4.mass

for i in range(len(all_species)):
    s = all_species[i]
    d[s.name+'_vol'] = d[s.name] / s.density

d['Org_vol'] = d['Org_AMS_STP']*1E-6/1.4

d['44_vol'] = d['mz44_AMS_STP']*1E-6/1.4

d['vc'] = d[['Org_vol', 'NH4NO3_vol', 'H2SO4_vol', 'NH4HSO4_vol', 'NH42SO4_vol']].sum(axis=1)

d.loc[d['vc']<=0, 'vc'] = np.nan

for i in range(len(all_species)):
    s = all_species[i]
    d[s.name+'_phi'] = d[s.name+'_vol'] / d['vc']
d['Org_phi'] = d['Org_vol'] / d['vc']


# Calculating chemical species including black carbon
d['BC_vol'] = d['BC_mass']*1E-9 / 2
d['vc_bc'] = d[['Org_vol', 'NH4NO3_vol', 'H2SO4_vol', 'NH4HSO4_vol', 'NH42SO4_vol', 'BC_vol']].sum(axis=1, min_count=5)

d.loc[d['vc_bc']<=0, 'vc_bc'] = np.nan

for i in range(len(all_species)):
    s = all_species[i]
    d[s.name+'_vf_bc'] = d[s.name+'_vol'] / d['vc_bc']
d['Org_vf_bc'] = d['Org_vol'] / d['vc_bc']

d['Org_vf_bc'] = d['Org_vol'] / d['vc_bc']

d['44_vf_bc'] = d['44_vol'] / d['vc_bc']
d['n44_vf_bc'] = d['Org_vf_bc'] - d['44_vf_bc']

d['BC_vf_bc'] = d['BC_vol'] / d['vc_bc']

# Screen out data with low AMS concentrations
d['AMS_tot_screen'] = d[['Org_AMS_STP', 'SO4_AMS_STP', 'NO3_AMS_STP', 'NH4_AMS_STP', 'Chl_AMS_STP']].sum(axis=1)
all_vf_keys = ['Org_phi', 'NH4NO3_phi', 'H2SO4_phi', 'NH4HSO4_phi', 'NH42SO4_phi',
                'NH4NO3_vf_bc',
                'H2SO4_vf_bc',
                'NH4HSO4_vf_bc',
                'NH42SO4_vf_bc',
                'Org_vf_bc',
                '44_vf_bc',
                'n44_vf_bc',
                'BC_vf_bc'] 
d.loc[d['AMS_tot_screen'] < 0.04, all_vf_keys] = np.nan

#%% Compile and resample 

# Set 'TF' to 1 where 'Time_Mid' is in the target dates
MetNav_out = MetNav.copy().resample('1 min', on='Time_Start').mean().reset_index()
MetNav_out = MetNav_out.sort_values(by='Time_Start').reset_index(drop=True)
MetNav_out['TF'] = MetNav_out['Time_Start'].dt.date.isin(target_dates).astype(int)
# Set 'TF' to nan where 'Time_Mid' is in the target dates
MetNav_out.loc[MetNav_out['Time_Start'].dt.date.isin(pd.to_datetime(['5/17/2024']).date), 'TF'] = np.nan
MetNav_out.loc[MetNav_out['Time_Start']<'2024-07-01', 'season'] = 'spr'
MetNav_out.loc[MetNav_out['Time_Start']>'2024-07-01', 'season'] = 'sum'

#%% Calculating volume weighted diameter

bns = ['UHSAS_Bin01',
 'UHSAS_Bin02',
 'UHSAS_Bin03',
 'UHSAS_Bin04',
 'UHSAS_Bin05',
 'UHSAS_Bin06',
 'UHSAS_Bin07',
 'UHSAS_Bin08',
 'UHSAS_Bin09',
 'UHSAS_Bin10',
 'UHSAS_Bin11',
 'UHSAS_Bin12',
 'UHSAS_Bin13',
 'UHSAS_Bin14',
 'UHSAS_Bin15',
 'UHSAS_Bin16',
 'UHSAS_Bin17',
 'UHSAS_Bin18',
 'UHSAS_Bin19',
 'UHSAS_Bin20',
 ]

bns_min = np.array([60, 69.06, 79.49, 91.5, 105.3, 121.2, 139.5, 160.6, 184.9, 212.8, 244.9, 281.9, 324.5, 373.6, 430, 494.9, 569.7, 655.7, 754.8, 868.8])
# geometric mean diameter
bns_mid = np.array([64.37, 74.1, 85.29, 98.17, 113, 130.1, 149.7, 172.3, 198.4, 228.3, 262.8, 302.5, 348.2, 400.8, 461.3, 531, 611.2, 703.5, 809.8, 932.1])
bns_max = np.array([69.06, 79.49, 91.5, 105.3, 121.2, 139.5, 160.6, 184.9, 212.8, 244.9, 281.9, 324.5, 373.6, 430, 494.9, 569.7, 655.7, 754.8, 868.8, 1000])
bns_dlogD = np.array([0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234, 0.01234])
sd_v = pd.DataFrame(columns=bns)
sd_v = np.pi/6*bns_mid**3*d[bns]

num = (bns_mid * sd_v[bns]).sum(axis=1).astype(float)
den = sd_v[bns].sum(axis=1).astype(float)

v_peak = np.divide(
    num, den,
    out=np.full(num.shape, np.nan, dtype=float),
    where=den != 0
)
d['v_peak'] = v_peak
d['v_close'] = np.abs(d['Dp'] - d['v_peak']) < 100

#%% Adding MLR when within 100 nm of volume weighted diameter

# add linear multivarible regression
vars = ['Org_phi', 'NH4NO3_phi', 'H2SO4_phi', 'NH4HSO4_phi', 'NH42SO4_phi']
cols = vars + ['RI', 'Dp']


df = d[d.v_close==True][cols].dropna()
df.loc[:, 'RI_pred_AMS'] = np.nan

# Prepare inputs
X = df[vars].values
y = df['RI'].values

# Fit linear regression
reg = LinearRegression(fit_intercept=False)  # your model has no intercept
reg.fit(X, y)

df['RI_pred_AMS'] = reg.predict(X)
d.loc[df.index, 'RI_pred_AMS'] = df['RI_pred_AMS']
d.loc[df.index, 'AMS_pred_abserr'] = np.abs(df['RI_pred_AMS']-df['RI'])
d.loc[df.index, 'AMS_pred_err'] = df['RI']-df['RI_pred_AMS']

#%% Add alt bins

# Add altitude binning
d['altitude_bin'] = pd.cut(d['GPS_Altitude'], bins=np.arange(0, d['GPS_Altitude'].max() + 500, 500))

#%% Calc f44

obv_zero = (d['mz44_AMS_STP'] >= 0) & (d['Org_AMS_STP'] > 0)

d.loc[obv_zero, 'f44'] = d.loc[obv_zero, 'mz44_AMS_STP'] / d.loc[obv_zero, 'Org_AMS_STP']
d.loc[d['f44']<=0, 'f44'] = np.nan
d['f44'] = pd.to_numeric(d['f44'])
d.loc[np.isinf(d['f44']), 'f44'] = np.nan

#%% Calc n_eff of organic aerosol

num = d['RI'] - (d[['NH4NO3_phi', 'H2SO4_phi', 'NH4HSO4_phi', 'NH42SO4_phi']]*np.array([1.554, 1.444, 1.494, 1.540])).sum(axis=1, skipna=False)
den = d['Org_phi']

d.loc[:,'n_oa'] = num/den
d['n_oa'] = pd.to_numeric(d['n_oa'])
d.loc[np.logical_not(np.isfinite(d['n_oa'])),'n_oa'] = np.nan
d.loc[(d['n_oa']>1.75),'n_oa'] = np.nan

#%% Export merged and resampled data

d.loc[d['Time_Mid']<'2024-07-01', 'season'] = 'spr'
d.loc[d['Time_Mid']>'2024-07-01', 'season'] = 'sum'


d.to_csv(output_path+'/merged_data.csv', index=False)
MetNav_out.to_csv(output_path+'/MetNav.csv', index=False)