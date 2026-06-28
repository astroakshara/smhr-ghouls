# Akshara SMHR Edits

Baseline used for comparison: `/Users/aksharaviswa/smhr-rpa`

Edited version: `/Users/aksharaviswa/smhr-ghouls/smhr-ghouls`

Code added for this version is marked in source files with `#Akshara edits` near the relevant added block or changed line.

## Summary of Changes

- Added live NLTE querying support through MPIA, with INSPECT fallback for supported non-Fe species.
- Added batching for MPIA requests so requests stay within the MPIA 99-line limit.
- Stored NLTE abundance information separately from LTE abundance information:
  - `nlte_delta`
  - `abundance_nlte`
  - filled NLTE abundance using the species-average correction when an individual line has no finite correction.
- Added NLTE controls to the Stellar Parameters tab:
  - `Use NLTE` mode for stellar-parameter solving.
  - `NLTE` query button for Fe I and Fe II stellar-parameter lines.
  - completion dialog showing how many eligible lines received NLTE offsets.
  - LTE and NLTE Fe abundance summaries.
  - Mashonkina et al. 2017 microturbulence display.
- Added NLTE controls to the Line Measurements tab:
  - `NLTE` query button for acceptable abundance lines.
  - completion dialog showing how many eligible lines received NLTE offsets, including source counts.
  - NLTE columns in the measurement table.
- Added global plotting controls in the Line Measurements tab for residual and flux y-axis limits.
- Added Review tab NLTE overlays:
  - NLTE triangle markers.
  - NLTE mean line.
  - NLTE linear fit line.
  - NLTE sigma band controlled by the Stellar Parameters `Sigma to plot` value.
- Added a Literature tab after Review:
  - reads `txt_table_recommended_v2021_MW.tsv`.
  - plots SAGA literature abundance comparisons.
  - overlays the current star in LTE and NLTE.
  - refreshes from the current Review/abundance summary.
- Updated exported abundance tables to include NLTE correction and final NLTE abundance columns.
- Added robust EW abundance measurement batching around MOOG `abfind` calls.
- Fixed MOOG output decoding for byte strings.
- Fixed the Summary Plot color error by using valid Matplotlib hex colors.

## Files with Akshara Edit Markers

- `smh/nlte.py`
- `smh/session.py`
- `smh/optimize_stellar_params.py`
- `smh/spectral_models/base.py`
- `smh/spectral_models/profile.py`
- `smh/gui/base.py`
- `smh/gui/stellar_parameters.py`
- `smh/gui/chemical_abundances.py`
- `smh/gui/review.py`
- `smh/gui/literature.py`
- `smh/gui/ui_mainwindow.py`
- `smh/radiative_transfer/moog/cog.py`
- `smh/smh_plotting.py`

## Radial Velocity and Normalization

The radial velocity tab code was not changed relative to `/Users/aksharaviswa/smhr-rpa`:

- `smh/gui/rv.py` matches the RPA version.

The normalization tab code was restored to match `/Users/aksharaviswa/smhr-rpa` exactly:

- `smh/gui/normalization.py` matches the RPA version.

This was verified with direct file comparisons against `/Users/aksharaviswa/smhr-rpa`.

