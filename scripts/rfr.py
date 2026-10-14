# %% Requirements

# Imports
from scripts.config import output_path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import RFECV
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
import seaborn as sns
from tqdm import tqdm
from collections import Counter

# Read and screen data
d = pd.read_csv(output_path+'/merged_data.csv', parse_dates=['Time_Start','Time_Stop','Time_Mid'])
d = d[d.TF == 0].reset_index(drop=True)
d = d[d.InletFlag_LARGE.isin([0])].reset_index(drop=True)
d = d[d.lwc<0.001].reset_index(drop=True)
d['date'] = d['Time_Start'].dt.date
d_far = d
d = d[d.v_close==True].reset_index(drop=True)




def run_ml(d_func, vars, ascending=True):

    y = d_func['RI']

    X = d_func[vars]

    regr = RandomForestRegressor(n_jobs=1, random_state=0)

    kf = KFold(
        n_splits=10,
        shuffle=True,
        random_state=1
    )


    print('Running regression...')
    y_pred = cross_val_predict(regr, X, y, cv=kf, n_jobs=-1)
    print('Regression complete.\nCalculating mean feature importance...')

    importances = []

    def fit_and_get_importance(train_idx):
        regr.fit(X.iloc[train_idx], y.iloc[train_idx])
        return regr.feature_importances_


    importances = Parallel(n_jobs=-1)(
        delayed(fit_and_get_importance)(train_idx)
        for train_idx, _ in kf.split(X)
    )

    importances = np.array(importances)

    fi = pd.DataFrame(
        {'mean':importances.mean(axis=0), 'std':importances.std(axis=0)},
        index=X.columns
    ).sort_values('mean', ascending=ascending)


    print('Done.')

    return y_pred, fi


def run_ml_flight(d_func, vars, ascending=True):

    y = d_func['RI']

    X = d_func[vars]

    regr = RandomForestRegressor(n_jobs=1, random_state=0)

    groups = d_func['date'].astype(str)

    # combine sparse days into merged group labels
    merge_map = {
    '2024-05-28': '2024-05-28+2024-05-30',
    '2024-05-30': '2024-05-28+2024-05-30',
    '2024-06-03': '2024-06-03+2024-06-05',
    '2024-06-05': '2024-06-03+2024-06-05'
    }
    groups = groups.replace(merge_map)
    
    gkf = LeaveOneGroupOut()

    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        test_dates = groups.iloc[test_idx].unique()
        print(f'Fold {i}: {len(test_idx)} test points, dates: {list(test_dates)}')

    print('Running regression...')
    y_pred = cross_val_predict(regr, X, y, cv=gkf, groups=groups, n_jobs=-1)
    print('Regression complete.\nCalculating mean feature importance...')

    importances = []

    def fit_and_get_importance(train_idx):
        regr.fit(X.iloc[train_idx], y.iloc[train_idx])
        return regr.feature_importances_


    importances = Parallel(n_jobs=-1)(
        delayed(fit_and_get_importance)(train_idx)
        for train_idx, _ in gkf.split(X, groups=groups)
    )

    importances = np.array(importances)

    fi = pd.DataFrame(
        {'mean':importances.mean(axis=0), 'std':importances.std(axis=0)},
        index=X.columns
    ).sort_values('mean', ascending=ascending)


    print('Done.')

    return y_pred, fi

def run_ml_flight_season(d_func, vars, ascending=True):

    y = d_func['RI']

    X = d_func[vars]

    regr = RandomForestRegressor(n_jobs=1, random_state=0)

    groups = d_func['season'].astype(str)
    
    gkf = LeaveOneGroupOut()

    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        test_dates = groups.iloc[test_idx].unique()
        print(f'Fold {i}: {len(test_idx)} test points, dates: {list(test_dates)}')

    print('Running regression...')
    y_pred = cross_val_predict(regr, X, y, cv=gkf, groups=groups, n_jobs=-1)
    print('Regression complete.\nCalculating mean feature importance...')

    importances = []

    def fit_and_get_importance(train_idx):
        regr.fit(X.iloc[train_idx], y.iloc[train_idx])
        return regr.feature_importances_


    importances = Parallel(n_jobs=-1)(
        delayed(fit_and_get_importance)(train_idx)
        for train_idx, _ in gkf.split(X, groups=groups)
    )

    importances = np.array(importances)

    fi = pd.DataFrame(
        {'mean':importances.mean(axis=0), 'std':importances.std(axis=0)},
        index=X.columns
    ).sort_values('mean', ascending=ascending)


    print('Done.')

    return y_pred, fi

p_names = {'RI':r'$n_\mathsf{eff}$','v_peak':r'$\overline{D}_{p,v}$', 'Dp':'$D_{p,\mathsf{DASH}}$', 'Org_vf_bc':'$𝒗_\mathsf{org}$', 'BC_vf_bc':'$𝒗_{\mathsf{BC}}$', 'Sc450_total':r'$\alpha_{s,450\,\mathsf{nm}}$', 'IR_Surf_Temp':'$T_\mathsf{surface}$', 'Total_Air_Temp':'$T_\mathsf{air}$', 'GPS_Altitude':'Altitude', 'NH4NO3_vf_bc':'$𝒗_{NH_4NO_3}$', 'H2SO4_vf_bc':'$𝒗_{H_2SO_4}$', 'NH4HSO4_vf_bc':'$𝒗_{NH_4HSO_4}$', 'NH42SO4_vf_bc':'$𝒗_{(NH_4)_2SO_4}$', '44_vf_bc':'$𝒗_{\mathsf{org} \,44}$', 'Wind_Speed':'Wind speed', 'k_dash':r'$𝜅_{\mathsf{DASH}}$', 'Latitude':'Lat', 'Static_Pressure':'$P$', 'fRH550_RH20to80':'f(RH)', 'NumFrac_Dust':r'$f_\mathsf{Dust}$', 'IntegV_gt1000nm_APS':r'$V_{> 1 \mu \mathsf{m}}$', 'Potential_Temp':r'$\theta$'}
sea = ['spr','sum']
sea_lab = ['Spring', 'Summer']



#%% Table S2 - Feature selection
# Note this cell will take some time to run. The other cells can run without it.
def run_ml_nested(d_func, candidate_vars):

    y = d_func['RI']
    X = d_func[candidate_vars]
    groups = d_func['date'].astype(str)
    
    merge_map = {
    '2024-05-28': '2024-05-28+2024-05-30',
    '2024-05-30': '2024-05-28+2024-05-30'
    }
    groups = groups.replace(merge_map)


    outer_cv = LeaveOneGroupOut()
    inner_cv = LeaveOneGroupOut()

   

    y_pred = np.full(len(y), np.nan)
    selected_features_per_fold = []

    outer_splits = list(outer_cv.split(X, y, groups=groups))
    for train_idx, test_idx in tqdm(outer_splits, desc='Outer folds'):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        rfr = RandomForestRegressor(random_state=0, n_jobs=1)

        # feature selection
        selector = RFECV(
            estimator=rfr,
            cv=inner_cv.split(X_train, y_train, groups=groups_train),
            n_jobs=-1
        )
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_]
        selected_features_per_fold.append(list(selected_features))


        # fit model
        rfr.fit(X_train[selected_features], y_train)
        y_pred[test_idx] = rfr.predict(X_test[selected_features])
        
    return y_pred, selected_features_per_fold

# Run feature selection

features = ['Dp', 'Org_vf_bc', 'v_peak', 'Sc450_total', 'fRH550_RH20to80', 'IR_Surf_Temp', 'Total_Air_Temp', 'Potential_Temp', 'GPS_Altitude', 'NH4NO3_vf_bc', 'H2SO4_vf_bc', 'NH4HSO4_vf_bc', 'NH42SO4_vf_bc', '44_vf_bc', 'BC_vf_bc', 'Wind_Speed', 'Latitude', 'Static_Pressure', 'NumFrac_Dust', 'IntegV_gt1000nm_APS']
y_pred, selected_features_per_fold = run_ml_nested(d, features)


n_folds = len(selected_features_per_fold)

# Start every feature at 0 so unselected features are included
freq = Counter()
for feat in features:
    freq[feat] = 0

# Count how many folds selected each feature
for fold_features in selected_features_per_fold:
    for feat in fold_features:
        freq[feat] += 1

# Print results (most selected first)
for feat, count in freq.most_common():
    print(f'{feat}: {count}/{n_folds} folds ({100*count/n_folds:.0f}%)')



# Save table

def save_feature_frequency_xlsx(selected_features_per_fold, features, output_path):
    n_folds = len(selected_features_per_fold)

    # Start every feature at 0 so unselected features are included
    freq = Counter()
    for feat in features:
        freq[feat] = 0

    # Count how many folds selected each feature
    for fold_features in selected_features_per_fold:
        for feat in fold_features:
            freq[feat] += 1

    # Build the table (most selected first)
    df = pd.DataFrame(freq.most_common(), columns=['Feature', 'Folds Selected'])
    df['Total Folds'] = n_folds
    df['Percent of Folds'] = df['Folds Selected'] / n_folds

    df.to_excel(output_path, index=False)

save_feature_frequency_xlsx(
    selected_features_per_fold,
    features,
    output_path=output_path + '/TableS2.xlsx'
)



#%% Run random forrest regression by variable

l_vars = [
    ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp', 'Static_Pressure', 'GPS_Altitude', 'H2SO4_vf_bc', '44_vf_bc', 'Wind_Speed', 'NumFrac_Dust', 'IntegV_gt1000nm_APS', 'fRH550_RH20to80', 'NH4NO3_vf_bc', 'NH4HSO4_vf_bc', 'NH42SO4_vf_bc'],
    ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp', 'Static_Pressure', 'GPS_Altitude', 'H2SO4_vf_bc', 'Wind_Speed', 'NumFrac_Dust', 'IntegV_gt1000nm_APS'],
    ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp']
    ]

yp = []
fi_l = []

for i in range(len(l_vars)):
    d[f'ml_{i}'] = np.nan
    d[f'ml_{i}'], fi = run_ml_flight(d, l_vars[i])
    fi_l.append(fi)

#%% Fig. 7 (must run cell above first)

fig,axs=plt.subplots(3,2,width_ratios=[1,3],figsize=(6,7.5), sharex='col')

for i in range(3):
    fi_pretty = fi_l[i].rename(
        p_names, axis="index")

    axs[i,0].barh(fi_pretty.index, fi_pretty.loc[:,'mean'], xerr=fi_pretty.loc[:,'std'], ecolor='red', color='black')

    ax_min, ax_max, = 1.46, 1.61
    axs[i,1].set_xlim([ax_min, ax_max])
    axs[i,1].set_ylim([ax_min, ax_max])
    axs[i,1].plot([ax_min, ax_max], [ax_min, ax_max], ls='--', c='black')

    ledgend_lab = ['Spring', 'Summer']
    for j in range(2):
        sea = ['spr', 'sum'][j]
        di = d[d.season==sea]
        axs[i,1].scatter(di['RI'], di[f'ml_{i}'], color=sns.color_palette("Set2")[j], s=2, alpha=.4, label=ledgend_lab[j])

    r2 = r2_score(d['RI'], d[f'ml_{i}'])
    axs[i,1].text(.02,.97,f'$R^2={r2:.2f}$',transform=axs[i,1].transAxes, va='top')

    # labels
    axs[i,1].set_ylabel('$n_{\mathsf{eff,RFR}}$')
axs[-1,1].set_xlabel('$n_{\mathsf{eff}}$')
axs[-1,0].set_xlabel('Feature importance')

axs[0,1].legend(bbox_to_anchor=(1,1), markerscale=3)
axs[0, 0].tick_params(axis='y', labelsize=7)

for i,ax in enumerate(axs.flatten()):
    if i%2:
        xy = (0.9,.03)
    else:
        xy = (0.7,.03)

    ax.annotate(['(a)','(b)', '(c)', '(d)', '(e)', '(f)'][i], xy=xy, xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='bottom')

fig.tight_layout()
plt.savefig(output_path+'/fig07.pdf', bbox_inches='tight')
#%% Run random forrest regression by season

vars = ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp']
yp = []
fi_l_sea = []
sea = ['spr','sum']
for i in range(2):
    mask = d.season == sea[i]
    di = d[mask]
    d.loc[mask,f'ml_{sea[i]}'] = np.nan
    d.loc[mask,f'ml_{sea[i]}'], fi = run_ml_flight(di, vars, ascending=False)
    fi_l_sea.append(fi)

# %% Fig. 8 (must run cell above first)
fig,axs=plt.subplots(2,2,height_ratios=[3,1],figsize=(6,4.75), sharey='row')

for i in range(2):
    fi_pretty = fi_l_sea[i].rename(p_names)

    axs[1,i].bar(fi_pretty.index, fi_pretty.loc[:,'mean'], yerr=fi_pretty.loc[:,'std'], ecolor='red', color='black')

    axs[1,i].set_xticklabels(axs[1,i].get_xticklabels(), rotation=55, ha='right', va='top', rotation_mode='anchor')

    ax_min, ax_max, = 1.46, 1.61
    axs[0,i].set_xlim([ax_min, ax_max])
    axs[0,i].set_ylim([ax_min, ax_max])
    axs[0,i].plot([ax_min, ax_max], [ax_min, ax_max], ls='--', c='black')

    
    mask = d.season == sea[i]
    di = d[mask]
    axs[0,i].scatter(di['RI'], di[f'ml_{sea[i]}'], color=sns.color_palette("Set2")[i], s=2, alpha=1)

    r2 = r2_score(di['RI'], di[f'ml_{sea[i]}'])
    axs[0,i].text(.02,.97,f'$R^2={r2:.2f}$',transform=axs[0,i].transAxes, va='top')

    #labels
    axs[0,i].set_title(sea_lab[i])
    axs[0,i].set_xlabel('$n_{\mathsf{eff}}$')
axs[0,0].set_ylabel('$n_{\mathsf{eff,RFR}}$')
axs[1,0].set_ylabel('Feature\nImportance')

for i,ax in enumerate(axs.flatten()):
    if i<2:
        xy = (0.03,1.03)
    else:
        xy = (.88,.7)
    ax.annotate(['(a)','(b)', '(c)', '(d)', '(e)', '(f)'][i], xy=xy, xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='bottom')

fig.tight_layout()
plt.savefig(output_path+'/fig08.pdf', bbox_inches='tight')

#%% Fig. 9

cor_vars = ['RI', 'Dp', 'Org_vf_bc', 'v_peak', 'Sc450_total', 'fRH550_RH20to80', 'IR_Surf_Temp', 'Total_Air_Temp', 'Potential_Temp', 'GPS_Altitude', 'NH4NO3_vf_bc', 'H2SO4_vf_bc', 'NH4HSO4_vf_bc', 'NH42SO4_vf_bc', '44_vf_bc', 'BC_vf_bc', 'Wind_Speed', 'k_dash', 'Latitude', 'Static_Pressure', 'NumFrac_Dust', 'IntegV_gt1000nm_APS']
matrix = d[cor_vars].corr().rename(p_names, axis=0).rename(p_names, axis=1)

fig,ax=plt.subplots(figsize=(6.5/13*17,5/13*17))
heatmap = sns.heatmap(matrix, annot=100*matrix, cmap="coolwarm", fmt=".0f", linewidths=0.5, vmin=-1, vmax=1, ax=ax)

# Set colorbar label
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation coefficient')

plt.savefig(output_path+'/fig09.pdf', bbox_inches='tight')
#%% Seasonal CV


l_vars = [
    ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp', 'Static_Pressure', 'GPS_Altitude', 'H2SO4_vf_bc', '44_vf_bc', 'Wind_Speed', 'NumFrac_Dust', 'IntegV_gt1000nm_APS', 'fRH550_RH20to80', 'NH4NO3_vf_bc', 'NH4HSO4_vf_bc', 'NH42SO4_vf_bc'],
    ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp', 'Static_Pressure', 'GPS_Altitude', 'H2SO4_vf_bc', 'Wind_Speed', 'NumFrac_Dust', 'IntegV_gt1000nm_APS'],
    ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp']
    ]

yp = []
fi_l = []

for i in range(len(l_vars)):
    d[f'ml_sea_{i}'] = np.nan
    d[f'ml_sea_{i}'], fi = run_ml_flight_season(d, l_vars[i], ascending=True)
    fi_l.append(fi)
#%% Fig. S4 (must run cell above first)

fig,axs=plt.subplots(3,2,width_ratios=[1,3],figsize=(6,7.5), sharex='col')

for i in range(3):
    fi_pretty = fi_l[i].rename(
        p_names, axis="index")

    axs[i,0].barh(fi_pretty.index, fi_pretty.loc[:,'mean'], xerr=fi_pretty.loc[:,'std'], ecolor='red', color='black')

    ax_min, ax_max, = 1.46, 1.61
    axs[i,1].set_xlim([ax_min, ax_max])
    axs[i,1].set_ylim([ax_min, ax_max])
    axs[i,1].plot([ax_min, ax_max], [ax_min, ax_max], ls='--', c='black')

    ledgend_lab = ['Spring', 'Summer']
    for j in range(2):
        sea = ['spr', 'sum'][j]
        di = d[d.season==sea]
        axs[i,1].scatter(di['RI'], di[f'ml_sea_{i}'], color=sns.color_palette("Set2")[j], s=2, alpha=.4, label=ledgend_lab[j])

    r2 = r2_score(d['RI'], d[f'ml_sea_{i}'])
    axs[i,1].text(.02,.97,f'$R^2={r2:.2f}$',transform=axs[i,1].transAxes, va='top')

    # labels
    axs[i,1].set_ylabel('$n_{\mathsf{eff,RFR}}$')
axs[-1,1].set_xlabel('$n_{\mathsf{eff}}$')
axs[-1,0].set_xlabel('Feature importance')

axs[0,1].legend(bbox_to_anchor=(1,1), markerscale=3)
axs[0, 0].tick_params(axis='y', labelsize=7)

for i,ax in enumerate(axs.flatten()):
    if i%2:
        xy = (0.9,.03)
    else:
        xy = (0.7,.03)

    ax.annotate(['(a)','(b)', '(c)', '(d)', '(e)', '(f)'][i], xy=xy, xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='bottom')

fig.tight_layout()
plt.savefig(output_path+'/figS4.pdf', bbox_inches='tight')


#%% Run random forrest regression by season without segmenting by D_p,v within 100 nm of DASH D_p

vars = ['Total_Air_Temp', 'Potential_Temp', 'Latitude', 'Dp', 'v_peak', 'BC_vf_bc', 'Sc450_total', 'Org_vf_bc', 'IR_Surf_Temp']
yp = []
fi_l_sea = []
sea = ['spr','sum']
for i in range(2):
    mask = d_far.season == sea[i]
    di = d_far[mask]
    d_far.loc[mask,f'ml_{sea[i]}'] = np.nan
    d_far.loc[mask,f'ml_{sea[i]}'], fi = run_ml_flight(di, vars, ascending=False)
    fi_l_sea.append(fi)

# %% Fig. S5 (must run cell above first)
fig,axs=plt.subplots(2,2,height_ratios=[3,1],figsize=(6,4.75), sharey='row')

for i in range(2):
    fi_pretty = fi_l_sea[i].rename(p_names)

    axs[1,i].bar(fi_pretty.index, fi_pretty.loc[:,'mean'], yerr=fi_pretty.loc[:,'std'], ecolor='red', color='black')

    axs[1,i].set_xticklabels(axs[1,i].get_xticklabels(), rotation=55, ha='right', va='top', rotation_mode='anchor')

    ax_min, ax_max, = 1.46, 1.61
    axs[0,i].set_xlim([ax_min, ax_max])
    axs[0,i].set_ylim([ax_min, ax_max])
    axs[0,i].plot([ax_min, ax_max], [ax_min, ax_max], ls='--', c='black')

    
    mask = d_far.season == sea[i]
    di = d_far[mask]
    axs[0,i].scatter(di['RI'], di[f'ml_{sea[i]}'], color=sns.color_palette("Set2")[i], s=2, alpha=1)

    r2 = r2_score(di['RI'], di[f'ml_{sea[i]}'])
    axs[0,i].text(.02,.97,f'$R^2={r2:.2f}$',transform=axs[0,i].transAxes, va='top')

    #labels
    axs[0,i].set_title(sea_lab[i])
    axs[0,i].set_xlabel('$n_{\mathsf{eff}}$')
axs[0,0].set_ylabel('$n_{\mathsf{eff,RFR}}$')
axs[1,0].set_ylabel('Feature\nImportance')

for i,ax in enumerate(axs.flatten()):
    if i<2:
        xy = (0.03,1.03)
    else:
        xy = (.88,.7)
    ax.annotate(['(a)','(b)', '(c)', '(d)', '(e)', '(f)'][i], xy=xy, xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='bottom')

fig.tight_layout()
plt.savefig(output_path+'/figS5.pdf', bbox_inches='tight')





#%% Fig. S6
cor_vars = ['RI', 'Dp', 'Org_vf_bc', 'v_peak', 'Sc450_total', 'fRH550_RH20to80', 'IR_Surf_Temp', 'Total_Air_Temp', 'Potential_Temp', 'GPS_Altitude', 'NH4NO3_vf_bc', 'H2SO4_vf_bc', 'NH4HSO4_vf_bc', 'NH42SO4_vf_bc', '44_vf_bc', 'BC_vf_bc', 'Wind_Speed', 'k_dash', 'Latitude', 'Static_Pressure', 'NumFrac_Dust', 'IntegV_gt1000nm_APS']


fig,axs=plt.subplots(1,3, figsize=(12,6), width_ratios=[1,1,.05])

for i in range(2):
    mask = d.season == sea[i]
    di = d[mask]

    matrix = di[cor_vars].corr().rename(p_names, axis=0).rename(p_names, axis=1)

    sns.heatmap(matrix, cbar=[0,1][i], cbar_ax=axs[2], annot=100*matrix, cmap="coolwarm", fmt=".0f", linewidths=0.5, vmin=-1, vmax=1, ax=axs[i])
    
    axs[i].set_title(['(a)', '(b)'][i], loc='left')
    axs[i].set_title(['Spring', 'Summer'][i])

axs[-1].set_ylabel('Correlation coefficient')
axs[1].tick_params(labelleft=False)
fig.tight_layout()
plt.savefig(output_path+'/figS6.pdf', bbox_inches='tight')
