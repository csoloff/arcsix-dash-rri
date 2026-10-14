#%% Requirements
from scripts.config import output_path
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

# Read
d = pd.read_csv(output_path+'/merged_data.csv', parse_dates=['Time_Start','Time_Stop','Time_Mid'])

# Data filtering
d = d[d.TF == 0].reset_index(drop=True)
d = d[d.InletFlag_LARGE.isin([0])].reset_index(drop=True)
d = d[d.lwc<0.001].reset_index(drop=True)
d['date'] = d['Time_Start'].dt.date

# Separate spring and summer data
dsp = d[d.season=='spr']
dsm = d[d.season=='sum']
# %% Seasonal stats

print('Spring:')
print(dsp.RI.describe())

print('\nSummer:')
print(dsm.RI.describe())

print('\nSeasonal diff:')
print(dsp.RI.mean()-dsm.RI.mean())

print('\nttest:')
t_stat, p_value = stats.ttest_ind(dsp.RI, dsm.RI, equal_var=False)

print(t_stat, p_value)


print('\nEffect Size:')
print(pg.compute_effsize(dsp.RI, dsm.RI, eftype='cohen'))   # Cohen's d



# Chemistry stats

# Org
print('\nttest:')
t_stat, p_value = stats.ttest_ind(dsp.Org_phi.dropna(), dsm.Org_phi.dropna(), equal_var=False)

print(t_stat, p_value)


print('\nEffect Size:')
print(pg.compute_effsize(dsp.Org_phi.dropna(), dsm.Org_phi.dropna(), eftype='cohen'))   # Cohen's d


# H2SO4
print('\nttest:')
t_stat, p_value = stats.ttest_ind(dsp.H2SO4_phi.dropna(), dsm.H2SO4_phi.dropna(), equal_var=False)

print(t_stat, p_value)


print('\nEffect Size:')
print(pg.compute_effsize(dsp.H2SO4_phi.dropna(), dsm.H2SO4_phi.dropna(), eftype='cohen'))   # Cohen's d

# NH42SO4
print('\nttest:')
t_stat, p_value = stats.ttest_ind(dsp.NH42SO4.dropna(), dsm.NH42SO4.dropna(), equal_var=False)

print(t_stat, p_value)

print('\nEffect Size:')
print(pg.compute_effsize(dsp.NH42SO4.dropna(), dsm.NH42SO4.dropna(), eftype='cohen'))   # Cohen's d


# %% Size stats

print('<=250')
for i, di in enumerate([dsp, dsm]):
    print(['Spring', 'Summer'][i])
    print(di[di.Dp<=250].RI.describe())
print('\n')

print('>250')
for i, di in enumerate([dsp, dsm]):
    print(['Spring', 'Summer'][i])
    print(di[di.Dp>250].RI.describe())
print('\n')


print('Stats')
for i, di in enumerate([dsp, dsm]):
    print(['Spring', 'Summer'][i])
    print(di.groupby('Dp')['RI'].describe())
print('\n')

print('\nttest:')
t_stat, p_value = stats.ttest_ind(dsp[dsp.Dp<=250].RI, dsp[dsp.Dp>250].RI, equal_var=False)

print(t_stat, p_value)

print('\nEffect Size:')
print(pg.compute_effsize(dsp[dsp.Dp<=250].RI, dsp[dsp.Dp>250].RI, eftype='cohen'))   # Cohen's d
#%% LMEM seasonal dependence treating flight day as independent

m0 = smf.mixedlm("RI ~ season", d, groups=d["date"])
r0 = m0.fit(reml=True)
print(r0.summary())
 
season_coef = r0.params["season[T.sum]"]
season_ci = r0.conf_int().loc["season[T.sum]"]
print(f"\nSeasonal difference: {season_coef:.4f}")
print(f"95% CI: [{season_ci[0]:.4f}, {season_ci[1]:.4f}]")

print(r0.cov_re)
print(np.sqrt(r0.cov_re.iloc[0, 0]))
icc = r0.cov_re.iloc[0,0] / (r0.cov_re.iloc[0,0] + r0.scale)


#%% LMEM seasonal dependence treating flight day as independent with C(Dp) + GPS_Altitude + Latitude + Longitude as additional fixed effects

m1 = smf.mixedlm("RI ~ season + C(Dp) + GPS_Altitude + Latitude + Longitude", d, groups=d["date"])
r1 = m1.fit(reml=True)
print(r1.summary())
 
adj_coef = r1.params["season[T.sum]"]
adj_ci = r1.conf_int().loc["season[T.sum]"]
print(f"\nSeasonal difference: {adj_coef:.4f}")
print(f"95% CI: [{adj_ci[0]:.4f}, {adj_ci[1]:.4f}]")

#%% LMEM spring size dependence

dsp.loc[dsp.Dp<=250, 'size_bin'] = 'a'
dsp.loc[dsp.Dp>250, 'size_bin'] = 'b'

m1 = smf.mixedlm("RI ~ size_bin + GPS_Altitude + Latitude + Longitude", dsp, groups=dsp["date"])
r1 = m1.fit(reml=True)
print(r1.summary())

adj_coef = r1.params["size_bin[T.b]"]
adj_ci = r1.conf_int().loc["size_bin[T.b]"]
print(f"\nAdjusted size difference: {adj_coef:.4f}")
print(f"95% CI: [{adj_ci[0]:.4f}, {adj_ci[1]:.4f}]")