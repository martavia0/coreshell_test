# -*- coding: utf-8 -*-
"""
Analysis of coreshell_test.nc (produced by coreshell_mixture.py).

1. Data summary         -> print dimensions/variables
2. Consistency checks   -> closure BABS = BABS_CS + BABS_EXT, species sum, NaN/negative scan
3. Bulk statistics      -> total absorption per wavelength, mixing-state and species
                           contributions, HAS_SHELL fractions, AAE (model)
4. Model vs observations-> per-wavelength MFB/RMSE/R2 against COBS, AAE model vs obs
5. Figures (PNG)        -> coreshell_test_babs_vs_wl.png, coreshell_test_size_bins.png,
                            coreshell_test_model_vs_obs.png, coreshell_test_timeseries.png,
                            coreshell_test_msd.png

Run:  python analyze_coreshell_test.py
"""
import os
import numpy as np
from netCDF4 import Dataset as ncd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import pandas as pd

# --- Files (resolved relative to this script so it runs from anywhere)
HERE = os.path.dirname(os.path.abspath(__file__))
F_MODEL = os.path.join(HERE, 'coreshell_test.nc')
F_OBS   = os.path.join(HERE, 'observed_physical_variables7.nc')
F_TXT   = os.path.join(HERE, 'Colocated_PM1_BCN.txt')

# --- Load model
mod = ncd(F_MODEL, 'r')
wls     = mod.variables['WL'][:].filled()                # (L,) nm
rc      = mod.variables['RC'][:].filled()                # (Sr,) um
qabs    = mod.variables['QABS'][:].filled()              # (N, L, Sr)
bcs     = mod.variables['BABS_CS'][:].filled()           # (N, L, Sr) m-1
bext    = mod.variables['BABS_EXT'][:].filled()          # (N, L, Sr) m-1
bext_sp = mod.variables['BABS_EXT_SP'][:].filled()       # (S, N, L, Sr) m-1
babs    = mod.variables['BABS'][:].filled()              # (N, L, Sr) m-1
has_sh  = mod.variables['HAS_SHELL'][:].filled()         # (N, Sr) 0/1
species = mod.getncattr('species').split(',')            # [org, so4, no3, nh4]
mod.close()

N, L, Sr = babs.shape
S = len(species)

# --- Load observations (optional: model-vs-obs section skipped if missing)
cobs = None
msd = {}                       # species -> (N, Sr) g m-3, for the MSD figure
if os.path.exists(F_OBS):
    obs = ncd(F_OBS, 'r')
    cobs = obs.variables['COBS'][:].filled()             # (F, L) m-1
    for _v, _n in [('RORG', 'org'), ('RSOO', 'soot'), ('RSO4', 'so4'),
                   ('RNO3', 'no3'), ('RNH4', 'nh4')]:
        if _v in obs.variables:
            msd[_n] = obs.variables[_v][:].filled()      # (N, Sr) g m-3
    obs.close()
    assert cobs.shape[0] == N, 'observation count differs from model'
else:
    print('NOTE: %s not found -> model-vs-obs section skipped' % F_OBS)

# --- Load Datetime (sample axis, from the colocated measurements file)
times = None
if os.path.exists(F_TXT):
    times = pd.to_datetime(
        pd.read_csv(F_TXT, sep='\t', usecols=['Datetime'])['Datetime'])
    if len(times) != N:
        print('NOTE: %s has %d rows (expected %d) -> index axis used'
              % (os.path.basename(F_TXT), len(times), N))
        times = None
x_ax = times.to_numpy() if times is not None else np.arange(N)

# seasonal phase along the annual cycle: +1 in mid-July (JJA, hot),
# -1 in mid-January (DJF, cold), ~0 at the equinoxes (mid-Apr/Oct)
season = None
if times is not None:
    doy = (times.dt.dayofyear
           + (times - times.dt.normalize()).dt.total_seconds() / 86400.0)
    season = np.asarray(np.cos(2.0 * np.pi * (doy - 196.0) / 365.25))
cmap = plt.get_cmap('coolwarm')
snorm = plt.Normalize(-1, 1)

# =====================================================================
# 1. Data summary
# =====================================================================
print('=' * 68)
print('1. DATA SUMMARY (%s)' % os.path.basename(F_MODEL))
print('=' * 68)
print('  N  (samples)      = %d' % N)
print('  L  (wavelengths)  = %d   %s nm' % (L, wls))
print('  Sr (size bins)    = %d   r_center %g - %g um' % (Sr, rc.min(), rc.max()))
print('  S  (ext. species) = %d   %s' % (S, ','.join(species)))

# =====================================================================
# 2. Consistency checks
# =====================================================================
print()
print('=' * 68)
print('2. CONSISTENCY CHECKS')
print('=' * 68)

# closure: BABS = BABS_CS + BABS_EXT
resid_cs = np.abs(babs - (bcs + bext)) / np.maximum(babs, 1e-30)
print('  max |BABS-(BABS_CS+BABS_EXT)|/BABS   = %.3e' % resid_cs.max())

# species sum: BABS_EXT = sum_S BABS_EXT_SP
resid_sp = np.abs(bext - bext_sp.sum(axis=0)) / np.maximum(bext, 1e-30)
print('  max |BABS_EXT-sum(BABS_EXT_SP)|/ext  = %.3e' % resid_sp.max())

n_nan = int(np.isnan(babs).sum() + np.isnan(bcs).sum() + np.isnan(bext).sum()
            + np.isnan(bext_sp).sum() + np.isnan(qabs).sum())
n_neg = int((babs < 0).sum() + (bcs < 0).sum() + (bext < 0).sum())
print('  NaN values in all BABS*,QABS         = %d' % n_nan)
print('  negative values in BABS*             = %d' % n_neg)
print('  QABS range                           = [%.4g, %.4g]' % (qabs.min(), qabs.max()))
print('  HAS_SHELL: %d/%d sample-bins shelled (%.1f %%)'
      % (int(has_sh.sum()), N * Sr, 100.0 * has_sh.mean()))
# =====================================================================
# 3. Bulk statistics
# =====================================================================
print()
print('=' * 68)
print('3. BULK STATISTICS (mean over samples)')
print('=' * 68)

babs_tot  = babs.sum(axis=2)          # (N, L) m-1, sum over size bins
bcs_tot   = bcs.sum(axis=2)           # (N, L)
bext_tot  = bext.sum(axis=2)          # (N, L)
mean_tot = babs_tot.mean(axis=0)      # (L,)
mean_cs  = bcs_tot.mean(axis=0)       # (L,)
mean_ext = bext_tot.mean(axis=0)      # (L,)
mean_sp  = bext_sp.mean(axis=(1, 3))  # (S, L), mean over samples and bins

hdr = '  %-8s %12s %10s %10s' % ('WL (nm)', 'BABS m-1', 'CS frac %', 'EXT frac %')
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
for l in range(L):
    tot = mean_tot[l]
    print('  %-8d %12.3e %10.1f %10.1f'
          % (wls[l], tot, 100.0 * mean_cs[l] / tot, 100.0 * mean_ext[l] / tot))

print()
print('  Externally mixed coating contribution (mean over samples, m-1):')
for s in range(S):
    print('    %-5s' % species[s] + ''.join(' %12.3e' % mean_sp[s, l] for l in range(L)))

# AAE = -d log10(sigma) / d log10(lambda), fitted per sample over all wavelengths
def aae(sigma):                       # (N, L) m-1 -> (N,)
    a = np.full(sigma.shape[0], np.nan)
    ok = (sigma > 0).all(axis=1)
    if ok.any():
        p = np.polyfit(np.log10(wls), np.log10(sigma[ok]).T, 1)   # slopes: (n_valid,)
        a[ok] = -p[0]
    return a

aae_model = aae(babs_tot)
fa = np.isfinite(aae_model)
print()
print('  AAE (model, %d valid samples):  mean %.2f  min %.2f  max %.2f'
      % (int(fa.sum()), aae_model[fa].mean(), aae_model[fa].min(), aae_model[fa].max()))

# per-bin statistics (used in figures)
shell_frac = has_sh.mean(axis=0)      # (Sr,)
mcs_b  = bcs.mean(axis=(0, 1))        # (Sr,) mean over samples and wavelengths
mext_b = bext.mean(axis=(0, 1))       # (Sr,)

# =====================================================================
# 4. Model vs observations
# =====================================================================
aae_obs = None
if cobs is not None:
    aae_obs = aae(cobs)
    print()
    print('=' * 68)
    print('4. MODEL VS OBSERVATIONS (COBS, %d samples)' % cobs.shape[0])
    print('=' * 68)
    hdr = '  %-8s %12s %12s %10s %10s %8s' % (
        'WL (nm)', 'mod mean', 'obs mean', 'MFB %', 'RMSE m-1', 'R2')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for l in range(L):
        m, o = babs_tot[:, l], cobs[:, l]
        ok = o > 0
        mfb  = 100.0 * ((m[ok] - o[ok]) / o[ok]).mean()
        rmse = np.sqrt(((m - o) ** 2).mean())
        r2   = 1.0 - ((m - o) ** 2).sum() / ((m - m.mean()) ** 2).sum()
        print('  %-8d %12.3e %12.3e %10.1f %10.3e %8.2f'
              % (wls[l], m.mean(), o.mean(), mfb, rmse, r2))
    fok = fa & np.isfinite(aae_obs)
    print('  AAE: obs mean %.2f | model mean %.2f | R = %.2f (n=%d)'
          % (aae_obs[fok].mean(), aae_model[fok].mean(),
             np.corrcoef(aae_model[fok], aae_obs[fok])[0, 1], int(fok.sum())))
# =====================================================================
# 5. Figures
# =====================================================================
print()
print('5. FIGURES')

# bin edges at the midpoints between neighbouring bin centers, so the bars
# tile the log x-axis edge-to-edge (drawn with align='edge'; a fixed
# data-unit width would overlap badly instead)
bedges = np.empty(Sr + 1)
bedges[0] = rc[0] - 0.5 * (rc[1] - rc[0])          # may be < 0, clipped by axis
bedges[1:-1] = 0.5 * (rc[:-1] + rc[1:])
bedges[-1] = rc[-1] + 0.5 * (rc[-1] - rc[-2])
bw = bedges[1:] - bedges[:-1]

# --- Fig 1: wavelength dependence -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].errorbar(wls, mean_tot, yerr=babs_tot.std(axis=0), fmt='o-', capsize=3,
               color='tab:red', label=r'model (mean $\pm$ 1$\sigma$)')
if cobs is not None:
    ax[0].plot(wls, cobs.mean(axis=0), 's--', color='k', label='observations (mean)')
ax[0].set_yscale('log')
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_ylabel(r'$B_{abs}$ (m$^{-1}$)')
ax[0].set_title('Total absorption by wavelength')
ax[0].legend()
ax[0].grid(alpha=0.3)

bottom = np.zeros(L)
parts = np.concatenate([mean_cs[None, :], mean_sp])    # (S+1, L)
labels = ['core-shell'] + species
colors = ['0.3', 'saddlebrown', 'red', 'blue', 'orange']    # core-shell, org, so4, no3, nh4
for i in range(S + 1):
    ax[1].bar(wls, parts[i], bottom=bottom, color=colors[i], label=labels[i], width=28)
    bottom += parts[i]
ax[1].set_yscale('log')
ax[1].set_xlabel('Wavelength (nm)')
ax[1].set_ylabel(r'mean $B_{abs}$ (m$^{-1}$)')
ax[1].set_title('Mixing-state and species contributions')
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis='y')
fig.tight_layout()
p1 = os.path.join(HERE, 'coreshell_test_babs_vs_wl.png')
fig.savefig(p1, dpi=200)
plt.close(fig)
print('  saved %s' % p1)

# --- Fig 2: size-bin profile -------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.bar(bedges[:-1], mcs_b,  width=bw, align='edge', color='0.3',
       label='core-shell')
ax.bar(bedges[:-1], mext_b, width=bw, align='edge', bottom=mcs_b,
       color='tab:orange', label='externally mixed')
ax.set_xscale('log')
ax.set_xlabel('Bin center radius (um)')
ax.set_ylabel(r'mean $\sigma_{abs}$ per bin (m$^{-1}$)')
ax.set_title('Size-bin profile (mean over samples and wavelengths)')
ax.grid(alpha=0.3, which='both')
ax2 = ax.twinx()
ax2.plot(rc, 100.0 * shell_frac, 'k^--', ms=5, label='HAS_SHELL fraction')
ax2.set_ylim(0, 105)
ax2.set_ylabel('% of samples with shell')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
fig.tight_layout()
p2 = os.path.join(HERE, 'coreshell_test_size_bins.png')
fig.savefig(p2, dpi=200)
plt.close(fig)
print('  saved %s' % p2)

# --- Fig 3: model vs observations (points coloured by season) ------------
if cobs is not None:
    def _sc(axp, x, y, s=8, mask=None):
        if mask is not None:
            x, y = x[mask], y[mask]
        kw = dict(s=s, alpha=0.5)
        if season is not None:
            kw.update(c=season if mask is None else season[mask],
                      cmap=cmap, norm=snorm, alpha=0.7)
        return axp.scatter(x, y, **kw)

    npan = L + 1
    ncol = 4
    nrow = int(np.ceil(npan / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    sc0 = None
    for l in range(L):
        axp = axes[l]
        m, o = babs_tot[:, l], cobs[:, l]
        c = _sc(axp, o, m)
        if sc0 is None:
            sc0 = c
        lo = o.max()
        axp.plot([o.min(), lo], [o.min(), lo], 'k-', lw=1, alpha=0.6)
        ok = o > 0
        mfb = 100.0 * ((m[ok] - o[ok]) / o[ok]).mean()
        r2  = 1.0 - ((m - o) ** 2).sum() / ((m - m.mean()) ** 2).sum()
        axp.set_title('%d nm   R$^2$=%.2f  MFB=%+.0f%%' % (wls[l], r2, mfb), fontsize=9)
        axp.set_xlabel('obs (m$^{-1}$)')
        axp.set_ylabel('model (m$^{-1}$)')
        axp.grid(alpha=0.3)
    axp = axes[L]
    fok = fa & np.isfinite(aae_obs)
    _sc(axp, aae_obs, aae_model, s=10, mask=fok)
    lo = max(aae_obs.max(), aae_model.max())
    axp.plot([0, lo], [0, lo], 'k-', lw=1, alpha=0.6)
    axp.set_title('AAE   R=%.2f' % np.corrcoef(aae_model[fok], aae_obs[fok])[0, 1],
                  fontsize=9)
    axp.set_xlabel('obs AAE')
    axp.set_ylabel('model AAE')
    axp.grid(alpha=0.3)
    fig.tight_layout()
    if season is not None:
        cb = fig.colorbar(sc0, ax=axes.tolist(), orientation='vertical',
                          pad=0.01, shrink=0.7)
        cb.set_ticks([-1.0, 0.0, 1.0])
        cb.set_ticklabels(['DJF', 'equinox', 'JJA'], fontsize=7)
        cb.set_label('Season (cold $\\rightarrow$ hot)', fontsize=8)
    p3 = os.path.join(HERE, 'coreshell_test_model_vs_obs.png')
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    print('  saved %s' % p3)

# --- Fig 4: timeseries at 3 key wavelengths (+ AAE) -----------------------
ikeys = [int(np.argmin(np.abs(wls - w))) for w in (370, 520, 880)]
has_aae = cobs is not None and aae_obs is not None
nrows = 4 if has_aae else 3
fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.5 * nrows), sharex=True)
# model line coloured by season along its length: drawn as short segments,
# each coloured with the phase of its left endpoint
def _sline(axp, x, y, svals):
    cols = cmap(snorm(svals[:-1]))
    for i in range(len(x) - 1):
        axp.plot(x[i:i + 2], y[i:i + 2], '-', lw=0.8, color=cols[i],
                 solid_capstyle='projecting')

for axp, l in zip(axes[:3], ikeys):
    if season is not None:
        _sline(axp, x_ax, babs_tot[:, l], season)
    else:
        axp.plot(x_ax, babs_tot[:, l], '-', color='tab:red', lw=0.8, label='model')
    if cobs is not None:
        axp.plot(x_ax, cobs[:, l], '.', color='k', ms=4, alpha=0.5,
                 label=('obs' if season is None else None))
    axp.set_yscale('log')
    axp.set_ylabel('%d nm (m$^{-1}$)' % wls[l])
    if season is not None:
        # coloured line has no single colour -> neutral legend entry
        h = [Line2D([0], [0], color='0.4', lw=1.2, label='model')]
        if cobs is not None:
            h.append(Line2D([0], [0], marker='.', color='w',
                            markerfacecolor='k', ms=5, label='obs'))
        axp.legend(handles=h, loc='upper right', fontsize=8)
    else:
        axp.legend(loc='upper right', fontsize=8)
    axp.grid(alpha=0.3, which='both')
    if times is not None:
        axp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
if has_aae:
    axp = axes[3]
    fok = np.isfinite(aae_model) & np.isfinite(aae_obs)
    if season is not None:
        _sline(axp, x_ax[fok], aae_model[fok], season[fok])
    else:
        axp.plot(x_ax[fok], aae_model[fok], '-', color='tab:red', lw=0.8,
                 label='model')
    axp.plot(x_ax[fok], aae_obs[fok], '.', color='k', ms=4, alpha=0.5,
             label=('obs' if season is None else None))
    axp.set_ylabel('AAE')
    if season is not None:
        h = [Line2D([0], [0], color='0.4', lw=1.2, label='model'),
             Line2D([0], [0], marker='.', color='w',
                    markerfacecolor='k', ms=5, label='obs')]
        axp.legend(handles=h, loc='upper right', fontsize=8)
    else:
        axp.legend(loc='upper right', fontsize=8)
    axp.grid(alpha=0.3)
axes[-1].set_xlabel('Datetime' if times is not None else 'Sample index')
fig.suptitle('Total absorption and AAE: model vs observations' if has_aae
             else 'Total absorption: model vs observations', y=1.0)
fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0, 1, 0.97))
if season is not None:
    sm = ScalarMappable(cmap=cmap, norm=snorm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical',
                      pad=0.02, shrink=0.6)
    cb.set_ticks([-1.0, 0.0, 1.0])
    cb.set_ticklabels(['DJF', 'equinox', 'JJA'], fontsize=7)
    cb.set_label('Season (cold $\\rightarrow$ hot)', fontsize=8)
p4 = os.path.join(HERE, 'coreshell_test_timeseries.png')
fig.savefig(p4, dpi=200)
plt.close(fig)
print('  saved %s' % p4)

# --- Fig 5: mass size distribution -----------------------------------------
# NB: stacked bars on a log y-axis are misleading (a segment's apparent
# height ~ log(top/bottom), so the bottom segment is hugely inflated), so
# each species is a line here, plus a 100%-stacked share panel (linear y)
# where the segment heights are the true fractions.
if msd:
    order = [s for s in ('soot', 'org', 'so4', 'no3', 'nh4') if s in msd]
    colors = {'soot': '0.3', 'org': 'saddlebrown', 'so4': 'red',
              'no3': 'blue', 'nh4': 'orange'}
    marks = {'soot': 'o', 'org': 's', 'so4': '^', 'no3': 'v', 'nh4': 'D'}
    mss = {n: msd[n].mean(axis=0) for n in order}     # (Sr,) g m-3, mean over samples
    tot = sum(mss[n] for n in order)
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(7.5, 6.8))
    for n in order:
        ax.plot(rc, mss[n], color=colors[n], marker=marks[n], ms=4, lw=1, label=n)
    ax.plot(rc, tot, 'k--', lw=1, label='total')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'mean mass per bin (g m$^{-3}$)')
    ax.set_title('Mass size distribution (mean over samples)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=3)
    fr = np.vstack([mss[n] for n in order])           # (S, Sr) g m-3
    fr = np.divide(fr, tot[None, :], out=np.zeros_like(fr),
                   where=tot[None, :] > 0) * 100.0    # % of bin mass
    bottom = np.zeros(Sr)
    for i, n in enumerate(order):
        axb.bar(bedges[:-1], fr[i], width=bw, align='edge', bottom=bottom,
                color=colors[n])
        bottom += fr[i]
    axb.set_xscale('log')
    axb.set_ylim(0, 100)
    axb.set_xlabel('Bin center radius (um)')
    axb.set_ylabel('% of bin mass')
    axb.set_title('Species share of bin mass (colors as above)')
    axb.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    p5 = os.path.join(HERE, 'coreshell_test_msd.png')
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    print('  saved %s' % p5)
else:
    print('NOTE: no RORG/RSOO/RSO4/RNO3/RNH4 in %s -> MSD figure skipped'
          % os.path.basename(F_OBS))

print()
print('Done.')
