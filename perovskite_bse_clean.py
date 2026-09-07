#!/usr/bin/env python3
"""
perovskite_bse_clean.py
=======================
Tight-binding + Bethe-Salpeter equation (BSE) for the excitons of 2D
Ruddlesden-Popper lead-halide perovskites.

Model
-----
1. Tight binding. A 16-orbital spinless sp3 Hamiltonian per inorganic Pb-I
   layer (Boyer-Richard convention) is promoted to 32 spinful orbitals with
   atomic spin-orbit coupling, and stacked into an n-layer slab as
   kron(I_n, H_32).
2. BSE. The two-particle problem is solved in the Tamm-Dancoff approximation
   for a zero-momentum (optical) exciton, with a static Rytova-Keldysh
   direct (attractive) kernel. Short-range electron-hole exchange is neglected.

Reference: Y. Cho and T. C. Berkelbach, J. Phys. Chem. Lett. 10, 6189 (2019).

Usage
-----
    python perovskite_bse_clean.py              # n=3, 30x30 grid, nv=nc=2
    python perovskite_bse_clean.py n Nk nv nc
    from perovskite_bse_clean import run; run(n=3, Nk=30)

Outputs (written to ./out_n{n}/)
    bands.png     - TB+SOC band structure along Gamma-X-M-Gamma
    spectrum.png  - BSE absorption spectrum
    stdout        - exciton energy, binding energy, radius
"""

import os, sys, time
import numpy as np
from numpy.linalg import eigh
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
#  Constants and calibrated parameters (MAPbI3)
# ==============================================================================

E2 = 14.4           # e^2 in eV*Angstrom

# Tight-binding parameters (Boyer-Richard convention), calibrated for MAPbI3.
PARAMS_MAPBI3 = (
    -9.21,   # eps_Bs  - Pb s on-site
     1.508,  # eps_Bp  - Pb p on-site
    -13.21,  # eps_Xs  - I s on-site
    -1.63,   # eps_Xp  - I p on-site
    -0.770,  # V_ss
     0.875,  # V_sp
     0.420,  # V_ps
    -1.575,  # V_pp_sigma
     0.560,  # V_pp_pi
     6.3,    # a  (Angstrom)
)
SOC_MAPBI3 = {"lambda_B": 1.30, "lambda_X": 0.90}   # spin-orbit strengths (eV)

NF_SC        = 26    # filled orbitals per single (one-layer) cell
EPS0_BSE     = 2.80  # BSE background dielectric (single fitted parameter)
EPS_I_KELD   = 6.1   # Keldysh inorganic epsilon (screening length)
EPS_ENV_KELD = 2.0   # Keldysh environment epsilon

# ==============================================================================
#  Tight-binding Hamiltonian
# ==============================================================================

def H_bulk(kx, ky, kz, p):
    """
    16x16 spinless TB Hamiltonian at (kx, ky, kz) with parameters p.
    Orbital order: Pb(s,px,py,pz) | I_x(s,px,py,pz) | I_y(...) | I_z(...)
    """
    (eBs, eBp, eXs, eXp, Vss, Vsp, Vps, Vpps, Vppp, a) = p
    H = np.zeros((16, 16), dtype=complex)

    C = [2*np.cos(k*a/2) for k in (kx, ky, kz)]   # f_alpha = 2 cos(k_alpha a/2)
    G = [2j*np.sin(k*a/2) for k in (kx, ky, kz)]  # g_alpha = 2i sin(k_alpha a/2)

    # On-site energies (factor 1/2 because H = h + h^dagger)
    H[0, 0] = eBs/2
    H[1:4, 1:4] = np.eye(3)*eBp/2
    for i in range(3):
        b = 4 + 4*i
        H[b, b] = eXs/2
        H[b+1:b+4, b+1:b+4] = np.eye(3)*eXp/2
    H[7, 7] = (eXp-0.1)/2
    H[11, 11] = (eXp-0.1)/2
    H[15, 15] = (eXp-0.1)/2

    # Pb-I hoppings for each bond direction alpha in {x, y, z}
    for s in range(3):
        b = 4 + 4*s          # orbital index of I_s block
        H[0,   b]     = Vss  * C[s]
        H[0,   b+s+1] = Vsp  * G[s]
        H[s+1, b]     = -Vps * G[s]
        H[s+1, b+s+1] = Vpps * C[s]
        for r in range(3):
            if r != s:
                H[r+1, b+r+1] = Vppp * C[s]

    return H + H.conj().T


def soc_p_block(lam):
    """6x6 L.S matrix for p-orbitals in (px^up,py^up,pz^up,px^dn,py^dn,pz^dn)."""
    i = 1j
    L = np.array([
        [0,  0, -i,  0,  0,  1],
        [0,  0,  0,  i, -1,  0],
        [i,  0,  0,  0,  0, -i],
        [0, -i,  0,  0, -i,  0],
        [0, -1,  0,  i,  0,  0],
        [1,  0,  i,  0,  0,  0],
    ], dtype=complex)
    return (lam/3.0)*L


def spinful_from_spinless(H16):
    """Lift 16x16 -> 32x32 by tensoring with the spin identity."""
    return np.kron(H16, np.eye(2))


def add_SOC(H32, soc):
    """Add atomic SOC to the Pb-p (rows 2-7) and I-p blocks of H32."""
    H = H32.copy()
    H[2:8, 2:8] += soc_p_block(soc["lambda_B"])
    for i in range(3):
        s = (4 + 4*i)*2 + 2
        H[s:s+6, s:s+6] += soc_p_block(soc["lambda_X"])
    return H


def H_slab(n, kx, ky, p, soc):
    """32n x 32n n-layer slab Hamiltonian at (kx, ky, kz=0)."""
    H32 = add_SOC(spinful_from_spinless(H_bulk(kx, ky, 0, p)), soc)
    return np.kron(np.eye(n), H32)


# ==============================================================================
#  k-grid and band-structure paths
# ==============================================================================

def build_kgrid(Nk, a):
    """
    Nk x Nk 2D zone-boundary k-grid, u = n/Nk. This samples the exact M-point
    (pi/a, pi/a), where the first bright exciton lives. Returns (Nk^2, 3).
    """
    G = 2*np.pi/a
    pts = []
    for n1 in range(Nk):
        for n2 in range(Nk):
            pts.append([n1/Nk*G, n2/Nk*G, 0.0])
    return np.array(pts)


def k_path(points, N=60):
    """Piecewise-linear k-path through high-symmetry points."""
    path = []
    for i in range(len(points)-1):
        for t in np.linspace(0, 1, N, endpoint=False):
            path.append(points[i]*(1-t) + points[i+1]*t)
    return np.array(path)


# ==============================================================================
#  BSE: transitions, direct kernel, solver
# ==============================================================================

def getr0(eps_i, eps_env, d):
    """Keldysh screening length r0 = d (eps_i - 1) / 2."""
    return d*(eps_i-1)/2.0


def _W_keldysh_mat(q, r0, eps0, a, Nk):
    """Element-wise Rytova-Keldysh potential W(q) with q=0 cell average."""
    W = 2*np.pi*E2/(q*(eps0 + r0*q) + 1e-8)
    # BZ-cell average at q=0 (Gygi-Baldereschi)
    Om = (2*np.pi)**2/(Nk*a**2); qc = np.sqrt(Om/np.pi)
    if r0 < 1e-10:
        W0 = 4*np.pi*E2/(qc*eps0)
    else:
        W0 = 4*np.pi*E2/(qc**2*r0)*np.log(1 + r0*qc/eps0)
    W[q < 1e-12] = W0
    return W


def buildtrans(kgrid, p, nf, nv, nc, soc):
    """
    Diagonalise H32 at every k-point and collect electron-hole transitions.

    Returns
    -------
    Cc    : list of eigenvectors (conduction), length Nk*nv*nc
    Cv    : list of eigenvectors (valence),    length Nk*nv*nc
    Ecv   : list of single-particle transition energies
    klist : list of k-vectors (one per transition)
    trans : list of (ik, v, c) index tuples
    """
    Et, Ct = [], []
    for k in kgrid:
        E, C = eigh(H_slab(1, k[0], k[1], p, soc))
        Et.append(E); Ct.append(C)
    Et = np.asarray(Et); Ct = np.asarray(Ct)

    Cc, Cv, Ecv, klist, trans = [], [], [], [], []
    for ik, k in enumerate(kgrid):
        for c in range(nf, nf+nc):
            for v in range(nf-nv, nf):
                trans.append((ik, v, c))
                Cc.append(Ct[ik, :, c])
                Cv.append(Ct[ik, :, v])
                Ecv.append(float(Et[ik, c] - Et[ik, v]))
                klist.append(k)
    return Cc, Cv, Ecv, klist, trans


def build_direct(Cc, Cv, kpts, r0, eps0, a, Nk):
    """
    Vectorised direct (attractive) kernel:
      K^d_{tt'} = -W(k-k') <C_c(k)|C_c(k')> <C_v(k')|C_v(k)>
    Returns (Ntrans, Ntrans) complex matrix.
    """
    Cm  = np.array(Cc); Vm = np.array(Cv)
    kxy = np.array(kpts)[:, :2]
    G   = 2*np.pi/a
    dk  = kxy[:, None, :] - kxy[None, :, :]
    dk  = ((dk + G/2) % G) - G/2
    q   = np.linalg.norm(dk, axis=2)
    W   = _W_keldysh_mat(q, r0, eps0, a, Nk)
    Occ = Cm.conj() @ Cm.T   # overlap <C_c(k)|C_c(k')>
    Ov  = Vm.conj() @ Vm.T   # overlap <C_v(k)|C_v(k')>
    return -W * Occ * Ov.T


def solve_bse(Cc, Cv, Ecv, klist, Nk, r0, eps0, a):
    """
    Build and diagonalise the Tamm-Dancoff BSE Hamiltonian
      H_BSE = diag(Ecv) + K^d / (Nk a^2)
    with the direct (attractive) Keldysh kernel only. Returns (eigvals, eigvecs).
    """
    Kd    = build_direct(Cc, Cv, klist, r0, eps0, a, Nk)
    Ecv_a = np.asarray(Ecv, dtype=float)
    H     = np.diag(Ecv_a) + Kd/(Nk*a**2)
    return eigh(H)


def exciton_props(Ex, A, Ecv, trans, a, kgrid, nb, outdir='.'):
    """
    Exciton binding energy and in-plane radius for bright state index nb.

    Returns (Eg, Eb, r) in eV, eV, Angstrom. The radius uses the BZ-wrapped
    second moment <(k-kM)^2> of the exciton envelope |A|^2.
    Also writes the envelope Wk(|k-kM|) to {outdir}/Wk.txt.
    """
    Eg  = np.min(Ecv)
    Eb  = Eg - Ex[nb]
    Wk  = np.zeros(kgrid.shape[0])
    for S, (ik, v, c) in enumerate(trans):
        Wk[ik] += np.abs(A[S, nb])**2
    Wk /= Wk.sum()

    kM  = np.array([np.pi/a, np.pi/a])
    G   = 2*np.pi/a
    dk  = kgrid[:, :2] - kM[None, :]
    dk[:, 0] = ((dk[:, 0]+G/2) % G) - G/2
    dk[:, 1] = ((dk[:, 1]+G/2) % G) - G/2

    dkn = np.linalg.norm(dk, axis=1)
    np.savetxt(os.path.join(outdir, 'Wk.txt'),
               np.column_stack((dkn, Wk)),
               header='|k-kM|_invAngstrom    Wk_weight', fmt='%.6e')
    aBZ = 1.0/np.sqrt(np.sum(Wk*np.sum(dk**2, axis=1)))
    return Eg, Eb, aBZ*np.sqrt(1.5)


# ==============================================================================
#  Optical spectra
# ==============================================================================

def velocity_matrix_elements(kgrid, trans, Cc, Cv, p, n, soc, dk_fd=1e-5):
    """
    Momentum matrix elements p^cv(k) = <C_c(k)|dH/dk|C_v(k)> for every
    transition t=(ik, v, c), by finite difference (step dk_fd in 1/Angstrom).
    Returns pvc : (Ntrans, 2) complex array (px, py).
    """
    dHx = {}; dHy = {}
    for ik in sorted({t[0] for t in trans}):
        kx, ky = kgrid[ik, 0], kgrid[ik, 1]
        dHx[ik] = (H_slab(1, kx+dk_fd, ky, p, soc) -
                   H_slab(1, kx-dk_fd, ky, p, soc))/(2*dk_fd)
        dHy[ik] = (H_slab(1, kx, ky+dk_fd, p, soc) -
                   H_slab(1, kx, ky-dk_fd, p, soc))/(2*dk_fd)

    pvc = []
    for S, (ik, v, c) in enumerate(trans):
        pvc.append([np.vdot(Cc[S], dHx[ik] @ Cv[S]),
                    np.vdot(Cc[S], dHy[ik] @ Cv[S])])
    return np.array(pvc)


def bse_optical_me(A, pvc):
    """BSE optical matrix element P^(S) = sum_t A^(S)*_t p^cv(t)."""
    return A.conj().T @ pvc


def absorption_spectrum(E_x, P, Ex0, Nomega=1200, Nk=900,
                        omega_min=1.5, omega_max=3.0, eta=0.012):
    """
    Lorentzian-broadened absorption spectrum eps2(omega).

    Returns (omega, eps2) over [omega_min, omega_max].
    """
    omega = np.linspace(omega_min, omega_max, Nomega)
    osc   = np.abs(P[:, 0])**2 + np.abs(P[:, 1])**2
    eps2 = np.zeros(Nomega)
    for S in range(len(E_x)):
        dE = omega - E_x[S]
        eps2 += osc[S]*(eta/np.pi)/(dE**2 + eta**2)
    return omega, eps2/Nk


# ==============================================================================
#  Plotting helpers
# ==============================================================================

def _plot_bands(n, nf, p, soc, outdir):
    """TB+SOC band structure along Gamma-X-M-Gamma."""
    a  = p[-1]
    kG = np.array([0, 0, 0]); kX = np.array([np.pi/a, 0, 0])
    kM = np.array([np.pi/a, np.pi/a, 0])
    N  = 60
    kp = k_path([kG, kX, kM, kG], N=N)
    ticks = [0, N, 2*N, 3*N]; tlabels = ['G', 'X', 'M', 'G']

    print("  Computing band structure...", flush=True)
    # Slab bands = n copies of the single-layer H32 spectrum at each k.
    E = []
    for kx, ky in kp[:, :2]:
        ev = np.linalg.eigvalsh(add_SOC(spinful_from_spinless(H_bulk(kx, ky, 0, p)), soc))
        E.append(np.sort(np.concatenate([ev]*n)))
    E = np.array(E)

    Ef = (E[:, nf-1].max() + E[:, nf].min())/2
    Eg = E[:, nf].min() - E[:, nf-1].max()

    nb = min(8, n*32-nf); irange = list(range(nf-nb, nf+nb))
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in irange:
        ax.plot(E[:, i] - Ef, 'k-', lw=0.8)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    for x in ticks:
        ax.axvline(x, color='gray', lw=0.4)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels)
    ax.set_ylabel('E - E_mid  (eV)')
    ax.set_title(f'MAPbI3 n={n}  TB+SOC  Eg={Eg*1e3:.0f} meV')
    ax.set_ylim(-3, 6)
    plt.tight_layout()
    fn = os.path.join(outdir, 'bands.png')
    plt.savefig(fn, dpi=150); plt.close()
    print(f"  -> {fn}")
    return Eg


def _plot_spectrum(Ex, Eg, Eb, r, P, om, eps, n, outdir,
                   omega_min=1.5, omega_max=3.0):
    """Single-panel BSE absorption spectrum."""
    fig, ax = plt.subplots(figsize=(9, 5))
    col  = '#1f77b4'
    emax = eps.max() if eps.max() > 0 else 1.0
    ax.fill_between(om, 0, eps/emax, alpha=0.25, color=col)
    ax.plot(om, eps/emax, color=col, lw=1.5, label='TB-BSE')
    ax.axvline(Ex[0], ls='--', color=col, lw=1.0, alpha=0.8,
               label=f'E_x = {Ex[0]:.4f} eV')
    ax.axvline(Eg, ls=':', color='gray', lw=0.8, label=f'E_g = {Eg:.4f} eV')

    pmax = np.max(np.abs(P[:, 0])) or 1.0
    for i in range(len(Ex)):
        if not (omega_min <= Ex[i] <= omega_max):
            continue
        if np.abs(P[i, 0]) > 0.1:
            ax.plot([Ex[i], Ex[i]], [0, np.abs(P[i, 0])/pmax],
                    color='steelblue', ls='--', lw=1.2)
        else:
            ax.plot([Ex[i], Ex[i]], [0, 0.3], color='orange', ls='--', lw=0.8)

    ax.set_xlabel('Energy (eV)'); ax.set_ylabel('eps2 (normalised)')
    ax.set_title(f'TB-BSE  |  E_b = {Eb*1e3:.0f} meV,  '
                 f'sqrt<r2> = {r:.1f} A  (n={n})', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(omega_min, omega_max)
    plt.tight_layout()
    fn = os.path.join(outdir, 'spectrum.png')
    plt.savefig(fn, dpi=150); plt.close()
    print(f"  -> {fn}")


# ==============================================================================
#  Main pipeline
# ==============================================================================

def run(n=3, Nk=30, nv=2, nc=2,
        params=PARAMS_MAPBI3, soc=SOC_MAPBI3,
        eps0=EPS0_BSE, outdir=None,
        omega_min=1.5, omega_max=3.0):
    """
    Full TB + BSE calculation (no GW, no exchange).

    Returns dict with keys: Eg, Ex, Eb, r.
    """
    a       = params[-1]
    nf      = NF_SC          # BSE uses single-layer (n=1) eigenvectors
    nf_slab = NF_SC*n        # n-layer Fermi index, used only for the band plot
    Nk2     = Nk**2
    r0      = getr0(EPS_I_KELD, EPS_ENV_KELD, a*n)

    if outdir is None:
        outdir = f'out_n{n}'
    os.makedirs(outdir, exist_ok=True)

    sep = '-'*60
    print(sep)
    print(f'  n={n}, k-grid={Nk}x{Nk} (M-point included), nv={nv}, nc={nc}')
    print(f'  nf={nf}, a={a} A, r0={r0:.3f} A, eps0={eps0}')
    print(sep)

    # 1. Band structure
    print('[1] Band structure')
    _plot_bands(n, nf_slab, params, soc, outdir)

    # 2. Transitions
    print('[2] Building transitions...')
    kgrid = build_kgrid(Nk, a)
    t0 = time.time()
    Cc, Cv, Ecv, klist, trans = buildtrans(kgrid, params, nf, nv, nc, soc)
    Cc_arr = np.array(Cc); Cv_arr = np.array(Cv)
    print(f'    Ntrans={len(Ecv)}, Eg={np.min(Ecv)*1e3:.1f} meV '
          f'({time.time()-t0:.1f}s)')

    # 3. BSE
    print('[3] Solving BSE (direct Keldysh kernel)...')
    t0 = time.time()
    pvc = velocity_matrix_elements(kgrid, trans, Cc_arr, Cv_arr, params, n, soc)
    Ex, A = solve_bse(Cc_arr, Cv_arr, Ecv, klist, Nk2, r0, eps0, a)
    P = bse_optical_me(A, pvc)
    nb = np.where(np.abs(P[:, 0]) > 0.1)[0][0]
    Eg, Eb, r = exciton_props(Ex, A, Ecv, trans, a, kgrid, nb, outdir=outdir)
    print(f'    bright exciton nb={nb}  ({time.time()-t0:.1f}s)')
    print(f'    Eg={Eg:.4f} eV  E_x={Ex[0]:.4f} eV  Eb={Eb*1e3:.1f} meV  r={r:.1f} A')

    np.save(os.path.join(outdir, 'Ex.npy'), Ex)
    np.save(os.path.join(outdir, 'A.npy'),  A)
    np.save(os.path.join(outdir, 'P.npy'),  P)

    # 4. Absorption spectrum
    print('[4] Absorption spectrum...')
    om, eps = absorption_spectrum(Ex, P, Ex[0], Nk=Nk2,
                                  omega_min=omega_min, omega_max=omega_max)
    _plot_spectrum(Ex, Eg, Eb, r, P, om, eps, n, outdir,
                   omega_min=omega_min, omega_max=omega_max)
    emax = eps.max() if eps.max() > 0 else 1.0
    np.savetxt(os.path.join(outdir, 'spectrum.txt'),
               np.column_stack((om, eps, eps/emax)),
               header='energy_eV    eps2_arb    eps2_normalised', fmt='%.6e')

    print(sep)
    print(f'  Eg={Eg:.4f} eV   E_x={Ex[0]:.4f} eV   '
          f'Eb={Eb*1e3:.1f} meV   r={r:.1f} A')
    print(sep)
    return dict(Eg=Eg, Ex=Ex[0], Eb=Eb, r=r)


# ==============================================================================
#  Entry point
# ==============================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    n         = int(args[0])        if len(args) > 0 else 3
    Nk        = int(args[1])        if len(args) > 1 else 30
    nv        = int(args[2])        if len(args) > 2 else 2
    nc        = int(args[3])        if len(args) > 3 else 2
    run(n=n, Nk=Nk, nv=nv, nc=nc)
