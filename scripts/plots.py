#%% Requirements

from scripts.config import output_path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch

# Read
d = pd.read_csv(output_path+'/merged_data.csv', parse_dates=['Time_Start','Time_Stop','Time_Mid'])

# Data filtering
d = d[d.TF == 0].reset_index(drop=True)
d = d[d.InletFlag_LARGE.isin([0])].reset_index(drop=True)
d = d[d.lwc<0.001].reset_index(drop=True)
dc = d[d.v_close] # separate dataset when D_p,v is within 100 nm of DASH D_p

# Separate spring and summer data
dsp = d[d.season=='spr']
dsm = d[d.season=='sum']

# Add MetNav data
MetNav = pd.read_csv(output_path+'/MetNav.csv', parse_dates=['Time_Start'])
MetNav = MetNav[MetNav.TF == 0].reset_index(drop=True)
MetNav_sp = MetNav[MetNav.season=='spr']
MetNav_sm = MetNav[MetNav.season=='sum']

# Global functions
def pick_day(d, date_str):
    return d[d.Time_Start.dt.date==pd.Timestamp(date_str).date()]
def percent_close(d, pred_key):
    d_local = d.dropna(subset=[pred_key]).reset_index(drop=True).copy()
    return 100*d_local.v_close.sum()/len(d_local)
def N(d, pred_key):
    d_local = d.dropna(subset=[pred_key]).reset_index(drop=True).copy()
    return len(d_local)
def peak_mean(d, pred_key):
    d_local = d.dropna(subset=[pred_key]).reset_index(drop=True).copy()
    return d_local.v_peak.median()

def add_counts(ax, d_func, bin_key, fontsize=9):
    # Count the number of points per bin and season
    counts = d_func.groupby([bin_key, 'season']).size().unstack(fill_value=0)
    for i, bin_label in enumerate(counts.index):
        # Get the x-position for this bin
        x_pos = i
        
        # Spring count and position
        spr_count = counts.loc[bin_label].get('spr', 0)
        if spr_count > 0:
            ax.text(x_pos - 0.2, ax.get_ylim()[1], f'{spr_count}', 
                    ha='center', va='bottom', color=palette[0], fontsize=fontsize)
        
        # Summer count and position
        sum_count = counts.loc[bin_label].get('sum', 0)
        if sum_count > 0:
            ax.text(x_pos + 0.2, ax.get_ylim()[1], f'{sum_count}', 
                    ha='center', va='bottom', color=palette[1], fontsize=fontsize)

def pick_time_range(d, start_time, end_time, stop_key='Time_Stop'):
    ts = d.Time_Start.dt.time
    te = d[stop_key].dt.time
    return d[(ts >= pd.to_datetime(start_time).time()) &
             (te <= pd.to_datetime(end_time).time())]

def subset(d, date, start_time, end_time, stop_key='Time_Stop'):

    di = pick_day(d, date)

    di = pick_time_range(di, start_time, end_time, stop_key=stop_key)

    return di

s = [
    'Org_vf_bc',
    'H2SO4_vf_bc',
    'NH4HSO4_vf_bc',
    'NH42SO4_vf_bc',
    'NH4NO3_vf_bc',
    'BC_vf_bc'
    ]

def get_vfs(d, start_time, end_time):

    di = pick_time_range(d, start_time, end_time)

    return di[s].mean()

palette = sns.color_palette("Set2")

#%% Fig. 1

index_ticks = [0]+list(d[d.Time_Mid.dt.date.diff() > pd.Timedelta('1 s')].index)
index_mids = []
for i in range(len(index_ticks)-1):
    index_mids.append(round(index_ticks[i]+(index_ticks[i+1]-index_ticks[i])/2))
index_mids.append(round(index_ticks[i]+(d.index[-1]-index_ticks[i])/2))



fig, axes = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [3.5, 1]}, figsize=(6.5, 4))
# Create a hexbin plot
ax = axes[0]
hb = ax.hexbin(d.index, d['RI'], gridsize=29, cmap='rainbow', mincnt=1)
ax.set_xlabel('Cumulative observations')
ax.set_ylabel(r'$n_\mathsf{eff}$', fontsize=12)
ax.axvline(dsp.index[-1]+1, c='black', alpha=1, ls='--')


date_labels = d['Time_Mid'].iloc[index_ticks].dt.strftime('%m-%d')
date_labels.iloc[1] = ''
date_labels.iloc[3] = ''
ax2 = ax.twiny()
ax2.set_xticks(index_ticks)
ax2.set_xticklabels(date_labels, rotation=90, ha='left', fontsize=9)
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Date')


ax.text(0.94, 0.95, '(a)', transform=ax.transAxes, verticalalignment='top')

ax = axes[1]
ax.text(0.80, 0.95, '(b)', transform=ax.transAxes, verticalalignment='top')

combined_data = np.concatenate([dsp['RI'], dsm['RI']])
bin_edges = np.linspace(min(combined_data), max(combined_data), num=29)


counts, _ = np.histogram(dsp['RI'], bins=bin_edges)
lns1 = ax.stairs(counts, bin_edges, label='Spring', orientation='horizontal', color=sns.color_palette("Set2")[0])


counts, _ = np.histogram(dsm['RI'], bins=bin_edges)


ax.stairs(counts, bin_edges, label='Summer', orientation='horizontal', alpha=.7, color=sns.color_palette("Set2")[1])


ax.legend(loc='lower right', frameon=False)

ax.set_xlabel('Total observations')


d['Date'] = d['Time_Mid'].dt.date
daily_mean = d.groupby('Date')['RI'].mean()


axes[0].plot(index_mids, daily_mean, color='black', lw=2, label='Daily Mean', alpha=1, ls='--')

plt.tight_layout()

cbar = fig.colorbar(hb, ax=axes, label='Observations', orientation='horizontal', pad=-0.12, shrink=0.40)


pos = cbar.ax.get_position()


cbar.ax.set_position([
    pos.x0 -.245,
    pos.y0 + .01,
    pos.width,
    pos.height
])
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')


axes[0].yaxis.set_minor_locator(AutoMinorLocator())
axes[0].tick_params(axis='y', which='minor', length=2)

plt.savefig(output_path+'/fig01.pdf', bbox_inches='tight')

#%% Fig. 2


warnings.filterwarnings("ignore", message="invalid value encountered in")
# Create figure
# Define projection

# setting up plot
proj = ccrs.LambertConformal(central_longitude=-45, central_latitude=75)
fig, ax = plt.subplots(2,2, figsize=(7, 5), subplot_kw={'projection': proj}, sharex=True, sharey=True)
ax[0,0].set_title('Spring')
ax[0,1].set_title('Summer')

# Set y-labels for the first column
# Manually add y-labels for the first column
ax[0, 0].text(-0.2, 0.5, '<1 km', va='center', ha='right', rotation='vertical', transform=ax[0, 0].transAxes)
ax[1, 0].text(-0.2, 0.5, '≥1 km', va='center', ha='right', rotation='vertical', transform=ax[1, 0].transAxes)

for l in range(2):
    
    for i in range(2):
        ax[l,i].set_extent([-90, -20, 72.936905, 85.922014], crs=ccrs.PlateCarree())
        ax[l,i].add_feature(cfeature.LAND, facecolor='green', alpha=0.2)
        gl = ax[l,i].gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                    linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = True
        #gl.right_labels = False
        gl.inline_labels = False
        
        gl.right_labels = False

        lons = [-80, -60, -40]
        lat_ref = 71      # latitude used only to compute the projected x-position
        y_pos = -0.05      # axes fraction, negative = below the plot

        for lon in lons:
            # Project (lon, lat_ref) from PlateCarree into the axes' own projection
            x_proj, y_proj = ax[l, i].projection.transform_point(lon, lat_ref, ccrs.PlateCarree())

            # Blend: x in data coords (already projected), y in axes-fraction coords
            trans = mtransforms.blended_transform_factory(ax[l, i].transData, ax[l, i].transAxes)

            ax[l, i].text(x_proj, y_pos, f"{-lon}°W",
                        transform=trans,
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize=8,
                        clip_on=False)

        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 20))
        gl.ylocator = mticker.FixedLocator(range(70, 91, 2))



    for i in range(2):
        di = [dsp, dsm][i]
        di = di[di.TF != 1]
        if l == 0:
            di = di[di.GPS_Altitude<1000]
        else:
            di = di[di.GPS_Altitude>=1000]
        MetNavi = [MetNav_sp, MetNav_sm][i]

        # Plot binned median RI as colored pixels

        lon_bins = np.linspace(-90, -20, 50)
        lat_bins = np.linspace(73, 86, 50)

        
        lon_idx = np.digitize(di['Longitude'], lon_bins) - 1
        lat_idx = np.digitize(di['Latitude'], lat_bins) - 1

        
        ri_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)


        for j in range(len(lat_bins)-1):
            for k in range(len(lon_bins)-1):
                mask = (lat_idx == j) & (lon_idx == k) & (di['RI'].notna())
                if mask.any():
                    ri_grid[j, k] = di.loc[mask, 'RI'].mean()

        
        lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)

        pc = ax[l,i].pcolormesh(lon_grid, lat_grid, ri_grid, cmap='seismic',
                        transform=ccrs.PlateCarree(), shading='auto', zorder=9999, vmin=1.51, vmax=1.57)
        
        ax[l,i].plot(MetNavi['Longitude'], MetNavi['Latitude'],
                color='grey', linewidth=0.8,
                transform=ccrs.PlateCarree(), zorder=0)
        
        label_lon = -68.75
        label_lat = 76.53

        ax[l,i].annotate('  Pituffik', xy=(label_lon, label_lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax[l,i]),
                xytext=(label_lon, label_lat), textcoords=ccrs.PlateCarree()._as_mpl_transform(ax[l,i]),
                horizontalalignment='left', verticalalignment='top')
        
        ax[l,i].plot(label_lon, label_lat, marker='^', color='white', markersize=8, markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder = 19999, alpha=.6)

    p_lat, p_long = 76.531111, -68.703056

    ax[l,0].text(-97, 80, 'Nares\nStrait↓', color='black', fontsize=9,transform=ccrs.PlateCarree(),horizontalalignment='left', verticalalignment='top', rotation=45)

    ax[l,1].text(-65, 86, 'Lincoln Sea\n↓', color='black', fontsize=9,
                transform=ccrs.PlateCarree(), horizontalalignment='center', verticalalignment='top')

for i,ax_i in enumerate(ax.flatten()):
    ax_i.annotate(['(a)','(b)','(c)','(d)'][i], xy=(0.03,.97), xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='top')


    ax_i.text(p_long+6, p_lat-1, 'Baffin\nBay', color='black', fontsize=9,
            transform=ccrs.PlateCarree(), horizontalalignment='right', verticalalignment='top')


cbar = plt.colorbar(pc, ax=ax, orientation='vertical', shrink=0.5, label='$n$')
cbar.set_label(label=r'$n_\mathsf{eff}$')


plt.savefig(output_path+'/fig02.pdf', dpi=300, bbox_inches='tight')

#%% Fig. 3

fig,axs = plt.subplots(1,2, sharey='all', sharex='all')

dl = [dsp, dsm]

for i in range(2):
    grouped_i = dl[i].groupby('altitude_bin', observed=False).agg(
        count=('RI', 'count'),
        median_RI=('RI', 'median'),
        q25_RI=('RI', lambda x: x.dropna().quantile(0.25) if x.dropna().size else np.nan),
        q75_RI=('RI', lambda x: x.dropna().quantile(0.75) if x.dropna().size else np.nan),
        median_org=('Org_phi', 'median'),
        q25_org=('Org_phi', lambda x: x.dropna().quantile(0.25) if x.dropna().size else np.nan),
        q75_org=('Org_phi', lambda x: x.dropna().quantile(0.75) if x.dropna().size else np.nan),

        median_theta=('Potential_Temp', 'median'),
        q25_theta=('Potential_Temp', lambda x: x.dropna().quantile(0.25) if x.dropna().size else np.nan),
        q75_theta=('Potential_Temp', lambda x: x.dropna().quantile(0.75) if x.dropna().size else np.nan)).reset_index()

    
    grouped_i['altitude_mid'] = [
    b.left if i == 0 else (b.right if i == len(grouped_i)-1 else b.mid)
    for i, b in enumerate(grouped_i['altitude_bin'])
    ]

    
    ax2 = axs[i].twiny()
    
    axs[i].set_zorder(10)
    ax2.set_zorder(1)
    axs[i].patch.set_visible(False)
    ax2.set_xlim([.2,.9])
    ax2.plot(grouped_i['median_org'], grouped_i['altitude_mid'], color='green', label='AMS')
    ax2.fill_betweenx(grouped_i['altitude_mid'], grouped_i['q25_org'], grouped_i['q75_org'], alpha=0.2, color='green')

    axs[i].plot(grouped_i['median_RI'], grouped_i['altitude_mid'], color='red', label='AMS', alpha=.7)
    axs[i].fill_betweenx(grouped_i['altitude_mid'], grouped_i['q25_RI'], grouped_i['q75_RI'], alpha=0.2, color='red')
    axs[i].scatter(dl[i].RI,dl[i].GPS_Altitude, s=1, color='red', alpha=.1, zorder=-1, rasterized=True)
    axs[i].set_xlim([1.5,1.6])

    axs[i].set_xlabel(r'$n_\mathsf{eff}$', color='red')
    ax2.set_xlabel(r'$𝒗_{\rm org}$', color='green')
    axs[i].set_ylim(0,6000)

    # Potential temperature gradient
    theta_valid = grouped_i[['altitude_mid', 'median_theta']].dropna()

    alt = theta_valid['altitude_mid'].to_numpy()
    theta = theta_valid['median_theta'].to_numpy()

    # Smooth theta before calculating derivative
    theta_smooth = gaussian_filter1d(theta, sigma=1)

    # dtheta/dz, converted from K/m to K/km
    dtheta_dz = np.gradient(theta_smooth, alt) * 1000

    # Plot
    ax3 = axs[i].twiny()
    ax3.spines["top"].set_position(("outward", 35))
    ax3.set_xlim([0, 8])

    ax3 = axs[i].twiny()
    ax3.spines["top"].set_position(("outward", 35))
    ax3.set_xlim([0, 8])

    ax3.plot(dtheta_dz, alt, color='purple')
    ax3.axvline(0, color='black', ls='--', lw=1)

    ax3.set_xlabel(r'$d\theta/dz$ [K km$^{-1}$]', color='purple')
    
    axs[i].text(0.05, 0.97, ['(a)','(b)'][i], transform=axs[i].transAxes, verticalalignment='top')
   
axs[0].set_ylabel('Altitude Above\nMean Sea Level [m]')
fig.tight_layout()

fig.savefig(output_path+'/fig03.pdf', dpi = 300)

#%% Fig. 4

fig,ax = plt.subplots(figsize=(6,3))
ax = sns.boxplot(data=d, x='Dp', y='RI', hue='season', showfliers=True, notch=True, palette='Set2')#["tab:blue", "tab:orange"])
ax.set_ylabel(r'$n_\mathsf{eff}$', fontsize=13)
ax.set_xlabel('$D_p$ [nm]', fontsize=13)
ax.tick_params(axis='x', rotation=45)

ax.tick_params(top=True, labeltop=False)
# Get the x-tick locations and labels
locs, labels = plt.xticks()
n_labels = []
for i in range(len(labels)):
    n_labels.append(int(float(labels[i].get_text())))
ax.set_xticklabels(n_labels)

# Count the number of points per bin and season
counts = d.groupby(['Dp', 'season']).size().unstack(fill_value=0)

for i, bin_label in enumerate(counts.index):
    # Get the x-position for this bin
    x_pos = i
    
    # Spring count and position
    spr_count = counts.loc[bin_label].get('spr', 0)
    if spr_count > 0:
        ax.text(x_pos - 0.2, ax.get_ylim()[1]+.005, f'{spr_count}', 
                ha='center', va='bottom', color=sns.color_palette("Set2")[0], rotation=75)
    
    # Summer count and position
    sum_count = counts.loc[bin_label].get('sum', 0)
    if sum_count > 0:
        ax.text(x_pos + 0.2, ax.get_ylim()[1]+.005, f'{sum_count}', 
                ha='center', va='bottom', color=sns.color_palette("Set2")[1], rotation=75)
leg = ax.legend(frameon=False)
leg.get_texts()[0].set_text('Spring')
plt.minorticks_on()
plt.grid(True, which='major', axis='y')
leg.get_texts()[1].set_text('Summer')


fig.savefig(output_path+'/fig04.pdf', bbox_inches='tight')

#%% Fig. 5

vars = ['Org_phi', 'NH4NO3_phi', 'H2SO4_phi', 'NH4HSO4_phi', 'NH42SO4_phi']

cols = vars + ['RI', 'Dp']
titles = ['170–180', '200', '250', '300–350', '450–500', '600–1000']
fig,axes = plt.subplots(2,3, sharex='all', sharey='all', figsize=(9,5))

Dps = [[170, 180], [200], [250], [300, 350], [450, 500], [600, 700, 800, 1000]]

for i in range(len(Dps)):
    ax = axes.flatten()[i]

    
    df = d[cols][d.Dp.isin(Dps[i])].dropna()
    

    # Prepare inputs
    X = df[vars].values
    y = df['RI'].values

    # Fit linear regression
    reg = LinearRegression(fit_intercept=False) 
    reg.fit(X, y)

    # Results
    print("Fitted parameters:", reg.coef_)

    # Predictions
    df['RI_pred'] = reg.predict(X)
    d.loc[:, 'RI_pred'] = np.nan
    d.loc[df.index, 'RI_pred'] = df['RI_pred']

    r = [d.RI_pred.min(),d.RI_pred.max()]
    r_p = [1.45, 1.625]
    for j in range(2):
        di = d[d.season == ['spr', 'sum'][j]]
        di_c = di[di.v_close==1]
        ax.scatter(di.RI, di.RI_pred, s=10, color=sns.color_palette("Set2")[j], label=['Spring', 'Summer'][j], alpha=1,zorder=0)
        ax.scatter(di_c.RI, di_c.RI_pred, s=10, color=sns.color_palette("Set2")[j], alpha=1, edgecolors='grey',zorder=1)

    
    ax.plot([1.45,1.65], [1.45,1.65], c='k', ls='--', label = 'one-to-one')
    
    ax.set_title(titles[i] + ' nm')
    
    
    'Org_phi', 'NH4NO3_phi', 'H2SO4_phi', 'NH4HSO4_phi', 'NH42SO4_phi'
    r2 = r2_score(df['RI'], df['RI_pred'])
    print("R2:", r2)

    ax.annotate(
    f'(NH$_4$)$_2$SO$_4={reg.coef_[4]:.2f}$\n'
    f'NH$_4$HSO$_4={reg.coef_[3]:.2f}$\n'
    f'NH$_4$NO$_3={reg.coef_[1]:.2f}$\n'
    f'H$_2$SO$_4={reg.coef_[2]:.2f}$\n'
    f'Org$={reg.coef_[0]:.2f}$\n',
    (.03, .97),
    xycoords='axes fraction',
    va='top',
    size=8,
    )

    ax.annotate(
    f'$N={N(d, 'RI_pred')}$\n'
    f'$R^2={r2:.2f}$\n'
    r'$\overline{D}_{p,v}$'f' = {round(peak_mean(d, 'RI_pred'))} nm\n'
    r'$P_\mathrm{<100\ nm}$'f' = {round(percent_close(d, 'RI_pred'))}%',
    (.97, .03),
    xycoords='axes fraction',
    size=8,
    ha='right',
    va='bottom'
    )

for i in range(len(Dps)):
    ax = axes.flatten()[i]
    ax.annotate(['(a)','(b)', '(c)', '(d)', '(e)', '(f)'][i], xy=(0.03,1.03), xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='bottom')

axes[0,2].legend(bbox_to_anchor=(1,1))
axes[0,0].set_xlim([1.45,1.65])
axes[0,0].set_ylim([1.45,1.65])
axes[1,1].set_xlabel('$n_\mathsf{eff}$')
for i in range(2):
    axes[i,0].set_ylabel('$n_\mathsf{eff,MLR}$')

fig.savefig(output_path+'/fig05.pdf')

#%% Fig. 6

fig,ax = plt.subplots(figsize=(5,3))
ax.set_ylim([1.45,1.75])
ax.scatter(d[d.season=='spr'].f44, d[d.season=='spr'].n_oa, s=3, color=sns.color_palette("Set2")[0], label='Spring')
ax.scatter(d[d.season=='sum'].f44, d[d.season=='sum'].n_oa, s=3, color=sns.color_palette("Set2")[1], label='Summer')

clean = d[['f44','n_oa']].dropna().sort_values('f44')

if len(clean) > 5:

    window = round(len(clean)*.1)
    
    y_smoothed = clean['n_oa'].rolling(window=window, center=True).median()
    y_75 = clean['n_oa'].rolling(window=window, center=True).quantile(.75)
    y_25 = clean['n_oa'].rolling(window=window, center=True).quantile(.25)
    
    ax.plot(clean['f44'], y_smoothed, color='black')
    ax.fill_between(clean['f44'], y_25, y_75, alpha=.25, color='black')
    
ax.set_xlim([0,.25])
plt.minorticks_on()
plt.grid(which='major')
ax.legend(markerscale=2.5)
ax.set_ylabel(r'$n_\mathsf{eff,lit,org}$')
ax.set_xlabel('$f_{44}$')
fig.savefig(output_path+'/fig06.pdf', bbox_inches='tight')

#%% Fig. 10

fig,axs = plt.subplots(2,figsize=(5.5,5))
BC_bins = np.linspace(0, 0.007, num=8)
BC_bins = np.delete(BC_bins, [-2])
BC_bins[-1]=0.07
bin_labels = [
    f"({100*BC_bins[i]:g}, {100*BC_bins[i+1]:g}]" 
    for i in range(len(BC_bins)-1)
]
dc['vBC_bin'] = pd.cut(dc['BC_vf_bc'], bins=BC_bins, labels=bin_labels)
sns.boxplot(data=dc, x='vBC_bin', y='AMS_pred_err', hue='season', showfliers=False, notch=True, ax=axs[0], palette='Set2')

axs[0].axhline(0, c='black', ls='--')
add_counts(axs[0], dc, 'vBC_bin')
axs[0].set_ylabel(r'$n_\mathsf{eff} - n_\mathsf{eff,MLR}$')
axs[0].set_xlabel(r'$v_\mathsf{BC}$ [%]')

axs[0].tick_params(axis='x', rotation=15)

labels = [item.get_text() for item in axs[0].get_xticklabels()]
labels[-1] = '$>0.5$'
axs[0].set_xticklabels(labels)

ax = axs[1]
axs[1].tick_params(axis='x', rotation=15)
V1_bins = np.logspace(-2, 1, num=7)
V1_bins = np.delete(V1_bins, [1,-2])
V1_bins[-1] = 100


exponents = np.linspace(-2, 1, num=7)
exponents = np.delete(exponents, [1,-2])
log_labels = [
    f"($10^{{{exponents[i]:g}}}$, $10^{{{exponents[i+1]:g}}}$]" 
    for i in range(len(exponents)-1)
]
log_labels[-2] = "($10^{-0.5}$, 1]"
log_labels[-1] = '$>1$'


dc['V1_bin'] = pd.cut(dc['IntegV_gt1000nm_APS'], bins=V1_bins, labels=log_labels)
sns.boxplot(data=dc, x='V1_bin', y='AMS_pred_err', hue='season', showfliers=False, notch=True, ax=axs[1], palette='Set2')


ax.axhline(0, color='black', ls='--')
add_counts(ax, dc, 'V1_bin')
axs[0].legend(bbox_to_anchor=(1,1), frameon=False)
axs[1].legend('', frameon=False)
fig.tight_layout()
ax.set_xlabel('$V_\mathsf{>1 \\ \mu m}$ [$\mathsf{\mu m^3}/$cm$^{3}$]')
axs[1].set_ylabel(r'$n_\mathsf{eff} - n_\mathsf{eff,MLR}$')
for i in range(2):
    axs[i].text(.93,.05, ['(a)','(b)'][i], transform=axs[i].transAxes)


handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, ['Spring', 'Summer'],
              bbox_to_anchor=(1, 1), frameon=False)

fig.savefig(output_path+'/fig10.pdf', bbox_inches='tight')

#%% Fig. 11
proj = ccrs.LambertConformal(central_longitude=-45, central_latitude=75)

fig, ax = plt.subplots(subplot_kw={'projection': proj})

ax.set_title('RF01\n2024-05-28', pad=15)
ax.coastlines()
ax.set_extent([-90, -20, 75, 85.922014], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='green', alpha=0.2, rasterized=True)

gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
            linewidth=0.5, color='gray', alpha=0.6, linestyle='--')

gl.right_labels = False
gl.inline_labels = False

lons = [-80, -60, -40]
lat_ref = 73
y_pos = -0.025

for lon in lons:
    # Project (lon, lat_ref) from PlateCarree into the axes' own projection
    x_proj, y_proj = ax.projection.transform_point(lon, lat_ref, ccrs.PlateCarree())

    # Blend: x in data coords (already projected), y in axes-fraction coords
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(x_proj, y_pos, f"{-lon}°W",
                transform=trans,
                horizontalalignment='center',
                verticalalignment='top',
                clip_on=False)

d1 = subset(d, '2024-05-28', '15:48', '17:58')#subset(d, '2024-06-05', '17:31', '18:21')

lons = d1.Longitude
lats = d1.Latitude
values = d1.RI

# Prepare line segments for coloring
points = np.array([lons, lats]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
lc = LineCollection(segments, cmap='coolwarm', norm=norm, transform=ccrs.PlateCarree(), linewidth=10, capstyle='round')


lc.set_array(values[:-1])


ax.add_collection(lc)


cbar = fig.colorbar(lc, ax=ax, orientation='vertical', label='$n$')


start_lon, start_lat = -70, 82
end_lon, end_lat = -76, 80
ax.annotate(
    '',
    xy=(end_lon, end_lat),
    xytext=(start_lon, start_lat),
    arrowprops=dict(facecolor='black', width=5, headwidth=12),
    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax)
)

p_lat, p_long = 76.531111, -68.703056

ax.scatter(p_long, p_lat, marker='X', c='white', edgecolor='black', s=250,
        transform=ccrs.PlateCarree(), zorder=1000)

# Add label with some offset so it doesn't overlap the marker
ax.text(p_long-3, p_lat-.9, 'Pituffik', color='black', fontsize=12,
        transform=ccrs.PlateCarree(), horizontalalignment='left', verticalalignment='top')

ax.text(p_long-10.5, p_lat-1.5, 'Baffin\nBay', color='black', fontsize=9,
        transform=ccrs.PlateCarree(), horizontalalignment='left', verticalalignment='top')

ax.text(-71, 83.9, 'Lincoln\nSea', color='black', fontsize=9,
        transform=ccrs.PlateCarree(), horizontalalignment='left', verticalalignment='top')

ax1 = fig.add_axes([0.595, 0.675, 0.15, 0.2])
ax1.set_xlabel('$v_{org}$')
ax1.set_ylabel('$n_\mathsf{eff}$')
ax1.set_yticks([1.525, 1.550, 1.575, 1.600])

x, y = d1.Org_vf_bc, d1.RI
cbar = ax1.scatter(x, y, c=d1.Dp, cmap='Set1', vmin=180, vmax=700, alpha=.75, edgecolor='black')

fig.colorbar(cbar, ax=ax1, label='$D_p$ [nm]', location='top')
# calc the trendline
idx = np.isfinite(x) & np.isfinite(y)
z = np.polyfit(x[idx], y[idx], 1)
p = np.poly1d(z)
r2 = r2_score(y[idx], p(x[idx]))
r = pearsonr(x[idx], y[idx])[0]


ax1.plot(x,p(x), label=f'$R^2={r2:.2f}$\n$r={r:.2f}$', c='black')
ax1.text(.1,.7,f'$r={r:.2f}$',transform=ax1.transAxes)




ax2 = fig.add_axes([0.20, 0.715, 0.2, 0.1])
ax2.set_xlabel('UTC Time')
ax2.set_ylabel('Alt [km]')


x = mdates.date2num(d1.Time_Start)
y = d1.GPS_Altitude/1000
c = d1.Org_phi  # Color by y-value, for example


points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create LineCollection with color array
lc = LineCollection(segments, cmap='coolwarm', linewidths=3, capstyle='round')
lc.set_array(c[:-1])  # Color each segment

cbar = ax2.add_collection(lc)

fig.colorbar(cbar, ax=ax2, label='$v_{org}$', location='top')
#plt.colorbar(lc, ax=ax, label='Color variable')


ax2.plot(d1.Time_Start, d1.GPS_Altitude/1000, alpha=0)
ax2.set_ylim([0,1])

# Format x-axis to show ticks every hour
ax2.xaxis.set_major_locator(mdates.HourLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


ax3 = fig.add_axes([0.43, 0.43, 0.2, 0.2])  # Adjust position and size as needed

#shift_lon, shift_lat  = -63.16407592413793, 81.62738341003134
shift_time = '2024-05-28 17:09'
shift_lon, shift_lat = d1[d1.Time_Start > shift_time][['Longitude', 'Latitude']].iloc[0]

ax.scatter(shift_lon, shift_lat, marker='x', c='red', s=250,
        transform=ccrs.PlateCarree(), zorder=1000, alpha=1)

ax2.axvline(pd.to_datetime(shift_time), c='red', linestyle='--')

colors = 'green', 'darkred', 'red', 'orange', 'blue', 'black'
sizes = get_vfs(d1, '0:00', shift_time).values

ax3.pie(sizes, colors=colors, wedgeprops={"alpha": 0.95})

print(sizes)
ax4 = fig.add_axes([0.3, 0.23, 0.2, 0.2])

sizes = get_vfs(d1, shift_time, '23:59').values

ax4.pie(sizes, colors=colors, wedgeprops={"alpha": 0.95})
print(sizes)

legend_elements = []

labels = ['Org', 'H$_2$SO$_4$', 'NH$_4$HSO$_4$', '(NH$_4$)$_2$SO$_4$', 'NH$_4$NO$_3$', 'BC']

for i in range(6):
    legend_elements.append(Patch(facecolor=colors[i], label=labels[i]))

ax.legend(handles=legend_elements, bbox_to_anchor=(.85, .02), frameon=False, loc='lower right', fontsize=9)


ax.text(
    0.89, 0.02,
    '(c)',
    transform=ax.transAxes,
    fontsize=15)

ax1.text(
    0.02, 0.05,
    '(b)',
    transform=ax1.transAxes)

ax2.text(
    0.01, 0.65,
    '(a)',
    transform=ax2.transAxes)

d1[d1.Dp<=250]['RI'].describe()
d1[d1.Dp>250]['RI'].describe()

fig.savefig(output_path+'/fig11.pdf', bbox_inches='tight', dpi=100)
#%% Fig. 12

proj = ccrs.LambertConformal(central_longitude=-45, central_latitude=75)

fig, ax = plt.subplots(subplot_kw={'projection': proj})

ax.coastlines()
ax.set_extent([-80, -60, 83.15, 84.4], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='green', alpha=0.2, rasterized=True)

gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
            linewidth=0.5, color='gray', alpha=0.6, linestyle='--')

gl.right_labels = False
gl.inline_labels = False


lons = [-75, -70, -65]
lat_for_labels = 84.4

for lon in lons:
    ax.text(lon, lat_for_labels, f"$-${-lon}°W",
            horizontalalignment='center',
            transform=ccrs.PlateCarree())

d1 = subset(d, '2024-05-31', '15:00', '16:50')

d1_sub = subset(d, '2024-05-31', '15:35:17', '16:41:30')


lons = d1_sub.Longitude
lats = d1_sub.Latitude
values = d1_sub.RI

points = np.array([lons, lats]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
lc = LineCollection(segments, cmap='coolwarm', norm=norm, transform=ccrs.PlateCarree(), linewidth=10, capstyle='round', alpha=.8)


lc.set_array(values[:-1])



ax.add_collection(lc)

cbar = fig.colorbar(lc, ax=ax, orientation='vertical', label='$n$')


start_lon, start_lat = lons.iloc[55], lats.iloc[55]-.12
end_lon, end_lat = lons.iloc[63], lats.iloc[63]-.12


arrow_outline = FancyArrowPatch(
    (start_lon, start_lat),
    (end_lon, end_lat),
    transform=ccrs.PlateCarree()._as_mpl_transform(ax),
    connectionstyle="arc3,rad=0.3",
    arrowstyle='Simple, tail_width=2, head_width=10, head_length=10',
    color='black',
    linewidth=4,
    alpha=1,
    zorder=4
)
ax.add_patch(arrow_outline)

# Create a curved arrow patch
arrow = FancyArrowPatch(
    (start_lon, start_lat),
    (end_lon, end_lat),
    transform=ccrs.PlateCarree()._as_mpl_transform(ax),
    connectionstyle="arc3,rad=0.3",
    arrowstyle='Simple, tail_width=2, head_width=10, head_length=10',
    color='yellow',
    linewidth=2,
    alpha=1,
    zorder=5
)


ax.add_patch(arrow)


ax.scatter(d1_sub.Longitude.iloc[0], d1_sub.Latitude.iloc[0], marker='v', c='green', s=150,
        transform=ccrs.PlateCarree(), zorder=1000, alpha=.5)
ax.scatter(d1_sub.Longitude.iloc[-1], d1_sub.Latitude.iloc[-1], marker='^', c='yellow', edgecolors='black', s= 150,
        transform=ccrs.PlateCarree(), zorder=5)


start_lon, start_lat = -68.5, 83.95
end_lon, end_lat = -65, 83.9
ax.annotate(
    '',
    xy=(end_lon, end_lat),
    xytext=(start_lon, start_lat),
    arrowprops=dict(facecolor='green', width=3, headwidth=11, edgecolor='green'),
    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax)
)


ax.text(d1_sub.Longitude.iloc[0], d1_sub.Latitude.iloc[0]+.07, 'Buoy-Q', color='black', fontsize=12,
        transform=ccrs.PlateCarree(), horizontalalignment='center', verticalalignment='bottom')



ax1 = fig.add_axes([0.545, .19, 0.2, 0.2])
ax1.set_xlabel('$v_{org}$')
ax1.set_ylabel(r'$n_\mathsf{eff}$')


x, y = d1.Org_phi, d1.RI
cbar = ax1.scatter(x, y, c=d1.Dp, cmap='Set1', vmin=180, vmax=700, alpha=.75, edgecolor='black')

fig.colorbar(cbar, ax=ax1, label='$D_p$ [nm]', location='top')
# calc the trendline
idx = np.isfinite(x) & np.isfinite(y)
z = np.polyfit(x[idx], y[idx], 1)
p = np.poly1d(z)
r2 = r2_score(y[idx], p(x[idx]))
r = pearsonr(x[idx], y[idx])[0]


ax1.plot(x,p(x), label=f'$R^2={r2:.2f}$\n$r={r:.2f}$', c='black')
ax1.text(.5,.2,f'$r={r:.2f}$',transform=ax1.transAxes)




ax2 = fig.add_axes([.125, .94, .75, .25])
ax2.set_xlabel('UTC Time')
ax2.set_ylabel('Alt [km]')

x = mdates.date2num(d1.Time_Start)
y = d1.GPS_Altitude/1000
c = d1.Org_phi


points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


lc = LineCollection(segments, cmap='coolwarm', linewidths=7, capstyle='round')
lc.set_array(c[:-1])

cbar = ax2.add_collection(lc)

fig.colorbar(cbar, ax=ax2, label='$v_{org}$', location='top')



ax2.plot(d1.Time_Start, d1.GPS_Altitude/1000, alpha=0)


ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

pies = []
xshifts=[0, .1, .185, .29, .415, .56, .662, .698]
yshifts=[0, 0, 0, 0, 0, 0, .12, -.04]
shift_times = ['0:00','15:25', '15:38:17', '15:50', '16:05', '16:20', '16:41:30','16:45','23:59']
acc=1
for i in range(len(shift_times)-1):
    pies.append(fig.add_axes([.12+xshifts[i], .98+yshifts[i], 0.07, 0.07]))  # Adjust position and size as needed

    if i < len(shift_times)-2:
        ax2.axvline(pd.to_datetime('2024-05-31 '+ shift_times[i+1]), c='black', linestyle='--', alpha=1)
        ax2.scatter(pd.to_datetime('2024-05-31 '+ shift_times[i+1])+pd.Timedelta('90s'), 1.9, marker=f'${acc}$', c='black', s=50,zorder=1000)

        sub = subset(d1, '2024-05-31', shift_times[i+1], shift_times[-1]).iloc[0]
        lon,lat = sub.Longitude, sub.Latitude
        if acc in [3,4,5]:
            ax.scatter(lon,lat, marker=f'${acc}$', c='black', s=50, transform=ccrs.PlateCarree(), zorder=5)
        acc += 1

    colors = 'green', 'darkred', 'red', 'orange', 'blue', 'black'
    sizes = get_vfs(d1, shift_times[i], shift_times[i+1]).values

    pies[i].pie(sizes, colors=colors, wedgeprops={"alpha": 0.95})

ax2.scatter(d1_sub.Time_Start.iloc[0], 1.25, marker='v', c='green', s=100, zorder=1000)
ax2.scatter(d1_sub.Time_Stop.iloc[-1], 1.25, marker='^', c='yellow', edgecolors='black', s= 100, zorder=1000)

legend_elements = []

labels = ['Org', 'H$_2$SO$_4$', 'NH$_4$HSO$_4$', '(NH$_4$)$_2$SO$_4$', 'NH$_4$NO$_3$', 'BC']

for i in range(6):
    legend_elements.append(Patch(facecolor=colors[i], label=labels[i]))

ax.legend(handles=legend_elements, bbox_to_anchor=(-.14, 1.16), frameon=True, loc='upper left', fontsize=7, ncol=3)


ax.text(
    0.9, 0.9,
    '(b)',
    transform=ax.transAxes, fontsize=15)

ax1.text(
    0.82, 0.05,
    '(c)',
    transform=ax1.transAxes)

ax2.text(
    0.01, 0.72,
    '(a)',
    transform=ax2.transAxes)

ax2.set_title('RF03 2024-05-31', loc= 'left', pad=40)
fig.savefig(output_path+'/fig12.pdf', bbox_inches='tight', dpi=100)

#%% Fig. 13

proj = ccrs.LambertConformal(central_longitude=-45, central_latitude=75)

fig, ax = plt.subplots(subplot_kw={'projection': proj})

ax.coastlines()
ax.set_extent([-60, -10, 82, 85.922014], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='green', alpha=0.2, rasterized=True)

gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
            linewidth=0.5, color='gray', alpha=0.6, linestyle='--')

gl.right_labels = False
gl.inline_labels = False


lons = [-60, -50, -40]
lat_for_labels = 84.5

for lon in lons:
    ax.text(lon, lat_for_labels, f"$-${-lon}°W",
            horizontalalignment='center',
            transform=ccrs.PlateCarree())
    

lons = [-10]
lat_for_labels = 83.4

for lon in lons:
    ax.text(lon, lat_for_labels, f"$-${-lon}°W",
            horizontalalignment='center',
            transform=ccrs.PlateCarree())

d1 = subset(d, '2024-06-06', '14:31', '17:00')

d1_sub = subset(d, '2024-06-06', '14:31', '16:45')

m_sub = subset(MetNav, '2024-06-06', '14:54', '17:00', stop_key='Time_Start')

lons = d1_sub.Longitude
lats = d1_sub.Latitude
values = d1_sub.RI


points = np.array([lons, lats]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
lc = LineCollection(segments, cmap='coolwarm', norm=norm, transform=ccrs.PlateCarree(), linewidth=10, capstyle='round', alpha=1)



lc.set_array(values[:-1])



ax.add_collection(lc)

cbar = fig.colorbar(lc, ax=ax, orientation='vertical', label='$n$')


start_lon, start_lat = lons.iloc[22], lats.iloc[22]+.3
end_lon, end_lat = lons.iloc[50], lats.iloc[50]+.3


arrow = FancyArrowPatch(
    (start_lon, start_lat),
    (end_lon, end_lat),
    transform=ccrs.PlateCarree()._as_mpl_transform(ax),
    connectionstyle="arc3,rad=0.3",
    arrowstyle='Simple, tail_width=2, head_width=10, head_length=10',
    color='black',
    linewidth=2,
    alpha=1,
    zorder=5
)


ax.add_patch(arrow)


ax.scatter(d1_sub.Longitude.iloc[-1], d1_sub.Latitude.iloc[-1], marker='^', c='yellow', edgecolors='black', s= 100,
        transform=ccrs.PlateCarree(), zorder=1000)



ax1 = fig.add_axes([0.545, .603, 0.2, 0.2])
ax1.set_xlabel('$v_{org}$')
ax1.set_ylabel(r'$n_\mathsf{eff}$')
#ax1.set_yticks([1.525, 1.550, 1.575, 1.600])

x, y = d1.Org_phi, d1.RI
cbar = ax1.scatter(x, y, c=d1.Dp, cmap='Set1', vmin=180, vmax=700, alpha=.75, edgecolor='black')

fig.colorbar(cbar, ax=ax1, label='$D_p$ [nm]', location='top')
# calc the trendline
idx = np.isfinite(x) & np.isfinite(y)
z = np.polyfit(x[idx], y[idx], 1)
p = np.poly1d(z)
r2 = r2_score(y[idx], p(x[idx]))
r = pearsonr(x[idx], y[idx])[0]


ax1.plot(x,p(x), label=f'$R^2={r2:.2f}$\n$r={r:.2f}$', c='black')
ax1.text(.5,.04,f'$r={r:.2f}$',transform=ax1.transAxes)




ax2 = fig.add_axes([.125, .94, .75, .25])
ax2.set_xlabel('UTC Time')
ax2.set_ylabel('Alt [km]')


x = mdates.date2num(d1.Time_Start)
y = d1.GPS_Altitude/1000
c = d1.Org_phi


points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap='coolwarm', linewidths=7, capstyle='round')
lc.set_array(c[:-1])

cbar = ax2.add_collection(lc)

fig.colorbar(cbar, ax=ax2, label='$v_{org}$', location='top')


ax2.plot(d1.Time_Start, d1.GPS_Altitude/1000, alpha=0)



ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

pies = []
xshifts=[.05, .19, .28, .323, .4, .53, .619, .68]
yshifts=[0, 0, 0, 0, 0, 0, 0, -.09]
shift_times = ['0:00', '15:15', '15:43', '15:47', '15:59', '16:15','16:43:33', '16:50', '23:59']
acc=1
for i in range(len(shift_times)-1):
    pies.append(fig.add_axes([.12+xshifts[i], 1.07+yshifts[i], 0.07, 0.07]))  # Adjust position and size as needed

    if i < len(shift_times)-2:
        ax2.axvline(pd.to_datetime('2024-06-06 '+ shift_times[i+1]), c='black', linestyle='--', alpha=1)
        ax2.scatter(pd.to_datetime('2024-06-06 '+ shift_times[i+1])+pd.Timedelta('120s'), .2, marker=f'${acc}$', c='black', s=50,zorder=1000)
        sub = subset(d1, '2024-06-06', shift_times[i+1], shift_times[-1]).iloc[0]
        lon,lat = sub.Longitude, sub.Latitude
        if acc < 7:
            ax.scatter(lon,lat, marker=f'${acc}$', c='black', s=50, transform=ccrs.PlateCarree(), zorder=5)
        acc += 1
    colors = 'green', 'darkred', 'red', 'orange', 'blue', 'black'
    sizes = get_vfs(d1, shift_times[i], shift_times[i+1]).values

    pies[i].pie(sizes, colors=colors, wedgeprops={"alpha": 0.95})


ax2.scatter(d1_sub.Time_Stop.iloc[-1], .5, marker='^', c='yellow', edgecolors='black', s= 100, zorder=1000)

legend_elements = []

labels = ['Org', 'H$_2$SO$_4$', 'NH$_4$HSO$_4$', '(NH$_4$)$_2$SO$_4$', 'NH$_4$NO$_3$', 'BC']

for i in range(6):
    legend_elements.append(Patch(facecolor=colors[i], label=labels[i]))

ax.legend(handles=legend_elements, bbox_to_anchor=(-.14, 1.17), frameon=True, loc='upper left', fontsize=7, ncol=3)


ax.text(
    0.01, 0.06,
    '(c)',
    transform=ax.transAxes, fontsize=15)

ax1.text(
    0.03, 0.82,
    '(b)',
    transform=ax1.transAxes)

ax2.text(
    0.01, 0.82,
    '(a)',
    transform=ax2.transAxes)

ax2.set_title('RF06 2024-06-06', loc= 'left', pad=40)

fig.savefig(output_path+'/fig13.pdf', bbox_inches='tight', dpi=100)

#%% Fig. S1

def lit_pred():
    d.loc[:,'RI_lit_pred'] = np.nan
    d.loc[:,'RI_lit_pred'] = (d[['Org_vf_bc', 'NH4NO3_vf_bc', 'H2SO4_vf_bc', 'NH4HSO4_vf_bc', 'NH42SO4_vf_bc', 'BC_vf_bc']]*np.array([1.555, 1.554, 1.444, 1.494, 1.540, 1.635])).sum(axis=1, skipna=False)
lit_pred()

titles = ['170–180', '200', '250', '300–350', '450–500', '600–1000']
fig,axes = plt.subplots(2,3, sharex='all', sharey='all', figsize=(9,5))
Dps = [[170, 180], [200], [250], [300, 350], [450, 500], [600, 700, 800, 1000]]

for i in range(len(Dps)):
    ax = axes.flatten()[i]
    df = d[['season', 'RI', 'RI_lit_pred', 'v_close']][d.Dp.isin(Dps[i])].dropna()

    r = [df.RI_lit_pred.min(),df.RI_lit_pred.max()]
    r_p = [1.45, 1.625]
    for j in range(2):
        di = df[df.season == ['spr', 'sum'][j]]
        di_c = di[di.v_close==1]
        ax.scatter(di.RI, di.RI_lit_pred, s=10, c=sns.color_palette("Set2")[j], label=['Spring', 'Summer'][j], alpha=1,zorder=0)
        ax.scatter(di_c.RI, di_c.RI_lit_pred, s=10, c=sns.color_palette("Set2")[j], alpha=1, edgecolors='grey',zorder=1)

    ax.plot([1.40,1.65], [1.40,1.65], c='k', ls='--', label = 'one-to-one')
    ax.set_title(titles[i] + ' nm')
    
    
    r2 = r2_score(df['RI'], df['RI_lit_pred'])
    print("R2:", r2)

    ax.annotate(
    r'$P_\mathrm{<100\ nm}$'f' = {round(percent_close(d[d.Dp.isin(Dps[i])], 'RI_lit_pred'))}%\n'
    r'$\overline{D}_{p,v}$'f' = {round(peak_mean(d[d.Dp.isin(Dps[i])], 'RI_lit_pred'))} nm\n'
    f'$R^2={r2:.2f}$\n'
    f'$N={N(d[d.Dp.isin(Dps[i])], 'RI_lit_pred')}$',
    (.03, .97),
    xycoords='axes fraction',
    size=8,
    ha='left',
    va='top'
    )

for i in range(len(Dps)):
    ax = axes.flatten()[i]
    ax.annotate(['(a)','(b)', '(c)', '(d)', '(e)', '(f)'][i], xy=(0.03,1.03), xycoords = 'axes fraction', horizontalalignment='left', verticalalignment='bottom')
    
axes[0,0].set_xlim([1.40,1.65])
axes[0,0].set_ylim([1.40,1.65])
axes[0,2].legend(bbox_to_anchor=(1,1))
axes[1,1].set_xlabel('$n_\mathsf{eff}$')
for i in range(2):
    axes[i,0].set_ylabel('$n_\mathsf{eff,lit}$')
fig.savefig(output_path+'/figS1.pdf', bbox_inches='tight')
#%% Fig. S2

fig,ax = plt.subplots(figsize=(5,2))
edges=np.linspace(0,.40,40)
d[d.season=='spr'].hist(column='f44', bins=edges, ax=ax, color=sns.color_palette("Set2")[0], label='Spring')
d[d.season=='sum'].hist(column='f44', bins=edges, ax=ax, alpha=.8, color=sns.color_palette("Set2")[1], label='Summer')
plt.xlim([0,.4])
plt.title('')

ax.set_xlabel(r'$f_{44}$')
ax.set_ylabel('Count')
ax.legend(frameon=False)
ax.grid(False)
fig.savefig(output_path+'/figS2.pdf')


#%% Fig. S3

# seperate panels for seasons
fig,axes = plt.subplots(1,2, figsize=(10,3), sharex='all', sharey='all')
for i in range(2):
    ax = axes[i]
    ax.set_ylim([1.45,1.75])
    di = [d[d.season=='spr'], d[d.season=='sum']][i]
    ax.scatter(di.f44, di.n_oa, s=3, color=sns.color_palette("Set2")[i], label='Spring')
    


    clean = di[['f44','n_oa']].dropna().sort_values('f44')

    if len(clean) > 5:
        
        window = round(len(clean)*.1)
        
        y_smoothed = clean['n_oa'].rolling(window=window, center=True).median()
        y_75 = clean['n_oa'].rolling(window=window, center=True).quantile(.75)
        y_25 = clean['n_oa'].rolling(window=window, center=True).quantile(.25)
        
        ax.plot(clean['f44'], y_smoothed, color='black')
        ax.fill_between(clean['f44'], y_25, y_75, alpha=.25, color='black')
        
    ax.set_xlim([0,.25])
    plt.minorticks_on()
    ax.grid()
    ax.set_xlabel('$f_{44}$')
    ax.set_title(['Spring', 'Summer'][i])
axes[0].set_ylabel(r'$n_\mathsf{eff,lit,org}$')

plt.tight_layout()

fig.savefig(output_path+'/figS3.pdf', bbox_inches='tight')