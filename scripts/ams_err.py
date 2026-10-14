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
#%% Run Monte Carlo simulation of ZSR mixing rule

rng = np.random.default_rng(0)
N_MC = 5000


sigma_indep = .5



rel_err = .5

SO4_v = d['SO4_AMS_STP'].values[None, :]
NO3_v = d['NO3_AMS_STP'].values[None, :]
NH4_v = d['NH4_AMS_STP'].values[None, :]
Org_v = d['Org_AMS_STP'].values[None, :]

n_rows = SO4_v.shape[1]


# independent factors: perturb between 0.5 and 1.5
def indep_factor():
    return rng.uniform(1 - rel_err, 1 + rel_err, size=(N_MC, n_rows))

SO4_p = SO4_v * indep_factor()
NO3_p = NO3_v * indep_factor()
NH4_p = NH4_v * indep_factor()
Org_p = Org_v * indep_factor()

SO4_m = SO4_p * 1e-6 / SO4.mass
NO3_m = NO3_p * 1e-6 / NO3.mass
NH4_m = NH4_p * 1e-6 / NH4.mass

NH4NO3_c  = NO3_m * NH4NO3.mass
H2SO4_c   = np.maximum(0, SO4_m - NH4_m + NO3_m) * H2SO4.mass
NH4HSO4_c = np.minimum(2*SO4_m - NH4_m + NO3_m, NH4_m - NO3_m) * NH4HSO4.mass
NH4HSO4_c = np.where(NH4HSO4_c < 0, 0, NH4HSO4_c)
NH42SO4_c = np.maximum(0, NH4_m - NO3_m - SO4_m) * NH42SO4.mass

NH4NO3_vol  = NH4NO3_c  / NH4NO3.density
H2SO4_vol   = H2SO4_c   / H2SO4.density
NH4HSO4_vol = NH4HSO4_c / NH4HSO4.density
NH42SO4_vol = NH42SO4_c / NH42SO4.density
Org_vol     = Org_p * 1e-6 / 1.4

vc = NH4NO3_vol + H2SO4_vol + NH4HSO4_vol + NH42SO4_vol + Org_vol
vc = np.where(vc <= 0, np.nan, vc)

phi = {
    'NH4NO3':  NH4NO3_vol  / vc,
    'H2SO4':   H2SO4_vol   / vc,
    'NH4HSO4': NH4HSO4_vol / vc,
    'NH42SO4': NH42SO4_vol / vc,
    'Org':     Org_vol     / vc,
}

phi_err = {k: np.nanstd(v, axis=0) for k, v in phi.items()}
typical_phi_err = {k: np.nanmean(v) for k, v in phi_err.items()}



species_list = ['NH4NO3', 'H2SO4', 'NH4HSO4', 'NH42SO4', 'Org']
# Print the typical error associated with each species
for spc in species_list:
    print(f'{spc}: {np.nanmean(phi_err[spc]):.2f}±{np.nanstd(phi_err[spc]):.2f}')