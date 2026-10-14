# Supplementary code repository
Used to reproduce results and figures for the manuscript "Airborne characterization of dry aerosol effective real refractive index in Arctic spring and summer."
## Setup
Once you clone this repository, you can use `uv` to create the virtual Python environment used in this analysis. Simply run the following command in the repository directory.
`uv sync`

### Datasets and versions
Download the following datasets into a single directory and specify the path in `config.py`.
- A-ATOFMS_R1
- DASH_R2
- FCDP_R1
- LARGE-AMS_R0
- LARGE-InletFlag_R1
- LARGE-MICROPHYSICAL_R0
- LARGE-OPTICAL_R0
- LARGE-SP2_R0
- MetNav_R0
- PUTLS-UHSAS_R1

You will also have to specify the path where the tables and figures generated from the scripts will export in `config.py`.

## Scripts
If you just want specific figures/tables generated, it is recommended to run the code in cells in a Python interactive. Cells are delimited by `#%%`. The `#%% Requirements` cell at the top of many of the scripts contains the necessary functions and data reading for the other cells to function properly.

### `config.py`
Where to specify the path where you store the ARCSIX data. The other scripts rely on importing these paths. The scripts assume the all the data is in the same directory with no subdirectories.

### `merge_data.py`
Script to merge datasets to the DASH-SP timestamps as well as perform calculations (e.g., ZSR mixing, kappa) which is used in other parts of the analysis. The script also compiles and exports a resampled MetNav dataset used for plotting flight tracks.
Note: This script must be run first before the other scripts will work properly because they rely on the exported merged data.

### `plots.py`
Generates all the figures in the manuscript and supplement except those used in the machine learning analysis. That is Fig. 1–6, Fig. 10–12, and Fig. S1–S3.

### `rfr.py`
Run feature selection, models, and generates all the figures/tables used in the machine learning analysis section (Fig. 7–9, Fig. S4–S5, Table S2).

### `stats.py`
Calculating stats for paper, including using a linear mixed effect model.

### `table3.py`
Generate flight by flight overview stats found in Table 3.

### `ams_err.py`
Run Monte Carlo simulation to estimate how AMS error propagates through the ZSR mixing rule.