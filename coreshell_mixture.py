import numpy as np
import PyMieScatt as pymie
from netCDF4 import Dataset as ncd

# --- Parameters
thick = 0.5          # shell thickness relative to core radius (rp = rc*(1+thick)), fixed
# What to do in bins where there is NOT enough coating mass to build the shell of that thickness:
#   'bare' -> soot core is treated as an uncoated sphere (Mie of the core only)
#   'zero' -> Qabs = 0 for that bin (bin ignored)
NO_SHELL_MODE = 'bare'

# --- Aerosol physical parameters
# densities, g/cm3 (org, soot, so4, no3, nh4)
dorg, dsoot, dso4, dno3, dnh4 = 1.4, 1.8, 1.77, 1.72, 1.75
# real refractive index
norg, nsoot, nso4, nno3, nnh4 = 1.53, 1.75, 1.52, 1.55, 1.54
# imaginary part
ksoot, kso4, kno3, knh4 = 0.79, 1e-8, 1e-8, 1e-8
korg = np.array([0.050, 0.025, 0.019, 0.013, 0.009, 0.004, 0.003])
wls = np.array([370, 470, 520, 590, 660, 880, 950])          # nm
rbin = np.array([0.001, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.3, 0.4, 0.5])
rbax = np.array([0.020, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
r_center = np.sqrt(rbin * rbax)       # geometric mean of bin edges, radius in um (assumed)

L = len(wls)
Sr = len(r_center)
assert len(korg) == L

# columns: 0=org, 1=soot, 2=so4, 3=no3, 4=nh4
d = np.array([dorg, dsoot, dso4, dno3, dnh4])             # (C,)
n = np.tile([norg, nsoot, nso4, nno3, nnh4], (L, 1))      # (L, C)
k = np.tile([0.0, ksoot, kso4, kno3, knh4], (L, 1))       # (L, C)
k[:, 0] = korg                                            # only organics depends on wavelength

# --- Observed physical variables
inp = ncd('./observed_physical_variables/observed_physical_variables7.nc', 'r')
rorg = inp.variables['RORG'][:, :].filled()   # (N, Sr) g m-3 mass per bin
rsoo = inp.variables['RSOO'][:, :].filled()
rso4 = inp.variables['RSO4'][:, :].filled()
rno3 = inp.variables['RNO3'][:, :].filled()
rnh4 = inp.variables['RNH4'][:, :].filled()
inp.close()
N = rorg.shape[0]
assert rorg.shape[1] == Sr


# --- Functions
def MixPhys(MMS, DDS, NNS, KKS, rul):
    """MMS [Ax] mass, DDS [Ax] density, NNS/KKS [Ax, L] refractive index."""
    MMS = np.asarray(MMS)
    DDS = np.asarray(DDS)
    NNS = np.asarray(NNS)
    KKS = np.asarray(KKS)
    L = NNS.shape[1]

    MAS = np.sum(MMS)
    if MAS > 0.0:
        VVS = MMS / DDS
        VVS = VVS / VVS.sum()                       # volume fractions
        DEN = np.average(DDS, weights=VVS)
        RRI, IRI = np.zeros(L), np.zeros(L)
        if rul == 'vmw':                            # homogeneously mixed
            for l in range(L):
                RRI[l] = np.average(NNS[:, l], weights=VVS)
                IRI[l] = np.average(KKS[:, l], weights=VVS)
    else:
        DEN, RRI, IRI = 1.0, np.ones(L), np.zeros(L)
    return MAS, DEN, RRI, IRI                       # (g m-3, g cm-3, -, -)


# --- Coating species (org, so4, no3, nh4)
idx = [0, 2, 3, 4]
rcoat = np.array([rorg, rso4, rno3, rnh4])          # (4, N, Sr)
dcoat = d[idx]                                      # (4,)
ncoat = n[:, idx].T                                 # (4, L)
kcoat = k[:, idx].T                                 # (4, L)

# --- Shell / external volume split
# Vc            = m_soot/d_soot
# Vshell_target = Vc*((1+t)^3 - 1)      volume needed for a shell of thickness t
# Vavail        = sum_i m_i/d_i         volume of coating material available
# Shell exists only if Vavail >= Vshell_target; then the shell takes exactly Vshell_target
# and the rest (Vext) is externally mixed. Otherwise: no core-shell, all coating is external.
vcore = rsoo / dsoot
vavail = np.sum(rcoat / dcoat[:, None, None], axis=0)
vshell_target = vcore * ((1.0 + thick) ** 3 - 1.0)

has_shell = (vcore > 0.0) & (vavail >= vshell_target)          # (N, Sr) bool
vshell = np.where(has_shell, vshell_target, 0.0)
vext = vavail - vshell
fshell = np.divide(vshell, vavail, out=np.zeros_like(vavail), where=vavail > 0)
rshell = rcoat * fshell[None, :, :]                 # (4, N, Sr) mass of each species in the shell

# --- Shell mixture properties
MAS_sh = np.zeros((N, Sr))
DEN_sh = np.zeros((N, Sr))
RRI_sh = np.zeros((N, Sr, L))
IRI_sh = np.zeros((N, Sr, L))
for i in range(N):
    for r in range(Sr):
        MAS_sh[i, r], DEN_sh[i, r], RRI_sh[i, r, :], IRI_sh[i, r, :] = MixPhys(
            rshell[:, i, r], dcoat, ncoat, kcoat, 'vmw')

m_shell = RRI_sh + 1j * IRI_sh                      # (N, Sr, L)
m_core = nsoot + 1j * ksoot 
# --- External coating (not in the shell): each species is an independent particle population
assert np.allclose(vshell + vext, vavail)           # volume conservation check
rext = rcoat * (1.0 - fshell)[None, :, :]           # (4, N, Sr) g m-3 of each species left outside
vext_sp = rext / dcoat[:, None, None]               # (4, N, Sr) cm3 m-3 of each species
assert np.allclose(vext_sp.sum(axis=0), vext)

# --- Qabs (PyMieScatt takes wavelength and DIAMETERS in nm)
Q = np.zeros((N, Sr, L))
for i in range(N):
    for r, rc in enumerate(r_center):
        if rsoo[i, r] <= 0.0:                       # no soot -> nothing to compute
            continue
        dcore = 2.0 * rc * 1e3                      # um radius -> nm diameter
        dshell = dcore * (1.0 + thick)              # fixed thickness
        for l, wl in enumerate(wls):
            if has_shell[i, r]:
                Q[i, r, l] = pymie.MieQCoreShell(m_core, m_shell[i, r, l], wl,
                                                 dcore, dshell, asDict=True)['Qabs']
            elif NO_SHELL_MODE == 'bare':
                Q[i, r, l] = pymie.MieQ(m_core, wl, dcore, asDict=True)['Qabs']
            # 'zero' -> leave Q = 0

# --- Qabs of each externally mixed species (pure homogeneous spheres at r_center)
# Depends only on species, size and wavelength, not on the measurement -> computed once.
species = ['org', 'so4', 'no3', 'nh4']
Q_sp = np.zeros((len(species), Sr, L))
for s_i in range(len(species)):
    for r, rc in enumerate(r_center):
        for l, wl in enumerate(wls):
            m_sp = ncoat[s_i, l] + 1j * kcoat[s_i, l]
            Q_sp[s_i, r, l] = pymie.MieQ(m_sp, wl, 2.0 * rc * 1e3, asDict=True)['Qabs']

# --- Absorption coefficient (m-1), per bin
# vcore, vext_sp are in cm3 m-3 (g m-3 / g cm-3) -> *1e-6 gives m3 m-3
rc_m = r_center * 1e-6                                           # m
Vp = (4.0 / 3.0) * np.pi * rc_m[None, :] ** 3                    # m3 per particle at r_center
n_core = vcore * 1e-6 / Vp                                       # (N, Sr) m-3
n_ext_sp = vext_sp * 1e-6 / Vp[None, :, :]                       # (4, N, Sr) m-3, = Vext_species/Vparticle
rp_m = rc_m[None, :] * np.where(has_shell, 1.0 + thick, 1.0)     # (N, Sr)

BABS_cs = n_core[:, :, None] * Q * (np.pi * rp_m ** 2)[:, :, None]                 # (N, Sr, L)
BABS_ext_sp = (n_ext_sp[:, :, :, None] * Q_sp[:, None, :, :]
               * (np.pi * rc_m ** 2)[None, None, :, None])                         # (4, N, Sr, L)
BABS_ext = BABS_ext_sp.sum(axis=0)                                                 # (N, Sr, L)
BABS = BABS_cs + BABS_ext

# --- Output
out = ncd('coreshell_test.nc', 'w', format='NETCDF4')
out.createDimension('N', N)
out.createDimension('L', L)
out.createDimension('Sr', Sr)

vwl = out.createVariable('WL', 'f4', ('L',));  vwl.units = 'nm';  vwl[:] = wls
vrc = out.createVariable('RC', 'f4', ('Sr',)); vrc.units = 'um'; vrc[:] = r_center

QABS = out.createVariable('QABS', 'f4', ('N', 'L', 'Sr'))
QABS.units = '1'
QABS.long_name = 'Core-shell absorption efficiency'
QABS[:, :, :] = Q.transpose(0, 2, 1)                # (N,Sr,L) -> (N,L,Sr)

out.createDimension('S', len(species))
out.species = ','.join(species)

QS = out.createVariable('QABS_SP', 'f4', ('S', 'L', 'Sr'))
QS.units = '1'
QS.long_name = 'Absorption efficiency of each externally mixed coating species'
QS[:, :, :] = Q_sp.transpose(0, 2, 1)                # (4,Sr,L) -> (4,L,Sr)

BS = out.createVariable('BABS_EXT_SP', 'f4', ('S', 'N', 'L', 'Sr'))
BS.units = 'm-1'
BS.long_name = 'Absorption coefficient of each externally mixed coating species'
BS[:, :, :, :] = BABS_ext_sp.transpose(0, 1, 3, 2)   # (4,N,Sr,L) -> (4,N,L,Sr)

for name, arr, ln in [('BABS_CS', BABS_cs, 'Absorption coefficient, core-shell (or bare core)'),
                      ('BABS_EXT', BABS_ext, 'Absorption coefficient, externally mixed coating'),
                      ('BABS', BABS, 'Total absorption coefficient per bin')]:
    v = out.createVariable(name, 'f4', ('N', 'L', 'Sr'))
    v.units = 'm-1'
    v.long_name = ln
    v[:, :, :] = arr.transpose(0, 2, 1)

HS = out.createVariable('HAS_SHELL', 'i1', ('N', 'Sr'))
HS.long_name = '1 if enough coating mass to build the shell of thickness thick'
HS[:, :] = has_shell.astype('i1')
out.close()
