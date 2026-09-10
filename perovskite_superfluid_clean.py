#!/usr/bin/env python3
"""
perovskite_superfluid_clean.py
==============================
Self-consistent excitonic-insulator / BCS superfluid populations for the
MAPbI3 tight-binding model, implementing Eqs. (11)-(16) of Perfetto &
Stefanucci, "Exceptional Excitons" (arXiv:2512.14392v1).

Uses the tight-binding model of perovskite_bse_clean.py (single-layer 32x32 H,
Keldysh-screened kernel). All outputs are written to ./out_n{n_layers}_superfluid/.

Single-band reduction: one valence band (top VB) and one conduction band
(bottom CB) of the single-layer 32x32 perovskite TB Hamiltonian, with the
n=3 Keldysh-screened kernel (r0 = getr0(eps_i, eps_env, a*n_layers)).

Exciton energy.  The condensate onset is the exciton energy E_x (Thouless),
the lowest eigenvalue of the equilibrium BSE Hamiltonian (see exciton_mode).
With SOC the band-edge states are Kramers doublets. The raw eigh eigenvectors of
a degenerate doublet are mixed arbitrarily and DISCONTINUOUSLY in k; that
scrambling corrupts the single-band overlaps <v k|v k'> and spuriously underbinds
the exciton (E_x ~ 2.11 eV, 65 meV -- a GAUGE ARTIFACT, not physics). _bands_at
removes it by S_z-diagonalising the doublet at each k (a smooth, physical
spin/J_z gauge; see _smooth_band), giving the correct E_x ~ 1.99 eV (binding
~185 meV) -- identical to the gauge-invariant 2v2c multiband
(perovskite_superfluid_multi.py). So the single band and the multiband agree once
the gauge is fixed; the doublet partners are degenerate J_z copies, not coupled.

For an input splitting Delta_mu = mu_c - mu_v we solve

    ( eps^r_vk - mu_v      Delta_k     ) ( phi^-_vk )        ( phi^-_vk )
    ( Delta_k^*        eps^r_ck - mu_c ) ( phi^-_ck )  = e^- ( phi^-_ck )     (13)

self-consistently, and read off the superfluid occupations

    f^sf_vk = |phi^-_vk|^2 ,   f^sf_ck = |phi^-_ck|^2 .

Kernel matrix elements (paper Eq. 5, W ~ V ~ Rytova-Keldysh):
  * Gap (Eq. 15):  V^{0}_{vcvc} = 0 (band orthogonality at same k), so
        Delta_k = -(1/(N a^2)) sum_k' W(k-k') <vk|vk'><ck'|ck> rho_vc(k')
  * Renormalized bands (Eq. 14): Hartree V^0 cancels since
        d(rho)_vv + d(rho)_cc = (|phi_v|^2-1) + |phi_c|^2 = 0 at every k',
        leaving the Fock/exchange self-energy (UNSCALED, bare 1v/1c)
        eps^r_ik = eps_ik - (1/(N a^2)) sum_{k',m} W(k-k') |<ik|mk'>|^2 drho_mk'
  * Lower eigenvalue (Eq. 16) and f^sf = |phi^-|^2 as stated below Eq. 16.

Chemical potentials (paper, below Eq. 16): max of equilibrium VB set to 0,
    mu_v = (Eg - Delta_mu)/2 ,  mu_c = (Eg + Delta_mu)/2 ,
so the (real) exciton energy is E_x = mu_c - mu_v = Delta_mu.

Usage
-----
    python perovskite_superfluid_clean.py             # scan Delta_mu, n=3, save data+plot
    python perovskite_superfluid_clean.py 41 0.30 cold   # notebook cold-start solver
    from perovskite_superfluid_clean import solve_superfluid, build_vc_bands
"""

import os
import numpy as np
from numpy.linalg import eigh
from perovskite_bse_clean import (PARAMS_MAPBI3, SOC_MAPBI3, H_slab, build_kgrid,
                                  getr0, _W_keldysh_mat, NF_SC, EPS0_BSE,
                                  EPS_I_KELD, EPS_ENV_KELD)

# Band degeneracy applied to the excitation density: each tracked (S_z gauge-fixed)
# valence/conduction band is 2-fold (Kramers) degenerate, so n carries a factor 2.
G_DEG = 2.0

# Excitation densities (cm^-2) for the ARPES panels and their txt exports.
ARPES_DENSITIES = (2e11, 6e11, 2e12, 3.2e12, 5e12, 7e12, 1e13)


def _n_label(nt):
    """Compact filename label for a density target, e.g. 2e11, 3.2e12, 1e13."""
    return ("%.1e" % nt).replace('+', '').replace('.0e', 'e')


# ---------------------------------------------------------------------------
#  Band structure: single-layer H32, top VB + bottom CB
# ---------------------------------------------------------------------------
# Spin operator S_z on the 32-orbital cell. H_slab builds the spinful cell as
# kron(H16, I2), i.e. ordering = 16 spatial (x) 2 spin with spin the fast index,
# so S_z = kron(I16, sigma_z/2). Used to GAUGE-FIX the Kramers doublets below.
_SZ32 = np.kron(np.eye(16), 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]]))


def _smooth_band(E, C, i, tol=1e-6):
    """Gauge-fixed eigenvector for band i.

    With SOC the band-edge states are 2-fold (Kramers) degenerate. eigh returns
    the two partners in an ARBITRARY, k-discontinuous mixture: the U(2) rotation
    within the degenerate subspace is unconstrained and jumps from k to k. That
    scrambling corrupts the single-band exciton overlaps <v k|v k'> (they collapse
    wherever the partner label flips) and spuriously UNDERbinds the exciton
    (65 meV instead of the physical 185 -- a gauge artifact, not physics).

    We remove the freedom by diagonalising S_z within the degenerate subspace and
    taking the highest-S_z partner: a smooth, physical (spin/J_z-labelled) choice.
    Non-degenerate bands are returned unchanged -- their residual U(1) phase does
    not affect the exciton eigenvalue."""
    grp = np.where(np.abs(E - E[i]) < tol)[0]        # degenerate multiplet of band i
    if len(grp) < 2:
        return C[:, i]
    D = C[:, grp]
    _, R = eigh(D.conj().T @ _SZ32 @ D)              # S_z-diagonalise within it
    return (D @ R)[:, -1]                            # highest-S_z partner (smooth)


def _bands_at(kxy, params, soc):
    """Top-VB / bottom-CB energies and (S_z gauge-fixed) eigenvectors at a list
    of k-points. The gauge fix (see _smooth_band) is essential for the single-band
    model: without it the Kramers doublet's raw eigh basis underbinds the exciton
    by ~3x."""
    n = len(kxy)
    eps_v = np.empty(n); eps_c = np.empty(n)
    Uv = np.empty((n, 32), complex); Uc = np.empty((n, 32), complex)
    iv, ic = NF_SC - 1, NF_SC
    for k in range(n):
        E, C = eigh(H_slab(1, kxy[k, 0], kxy[k, 1], params, soc))
        eps_v[k] = E[iv]; eps_c[k] = E[ic]
        Uv[k] = _smooth_band(E, C, iv)               # gauge-fixed (Kramers doublet)
        Uc[k] = _smooth_band(E, C, ic)
    return eps_v, eps_c, Uv, Uc


def find_band_edge(params=PARAMS_MAPBI3, soc=SOC_MAPBI3, Nk_scan=24):
    """Locate the (v->c) direct-gap minimum on a coarse full-BZ scan."""
    a = params[-1]
    kg = build_kgrid(Nk_scan, a)
    ev, ec, _, _ = _bands_at(kg[:, :2], params, soc)
    k0 = int(np.argmin(ec - ev))
    return kg[k0, :2]


def build_vc_bands(Nk=41, halfwidth=0.30, center=None,
                   params=PARAMS_MAPBI3, soc=SOC_MAPBI3):
    """
    Build a DENSE k-plaquette of side 2*halfwidth [1/A], Nk x Nk points,
    centred on the band edge (as in the paper's Methods: small window around
    the valley, not the full BZ). Diagonalise the single-layer 32x32 H and
    return the top VB / bottom CB, with the equilibrium VB maximum at zero.

    Returns kxy, eps_v, eps_c, Uv, Uc, Eg, dk_step, idx
      dk_step : plaquette spacing [1/A] (Delta k), used for the BZ-integral
                normalisation 1/(N a^2) -> (dk_step)^2/(2 pi)^2.
      idx     : integer plaquette indices centred at 0, for the torus wrap.
    """
    if center is None:
        center = find_band_edge(params, soc)
    ax = np.linspace(-halfwidth, halfwidth, Nk)
    dk_step = float(ax[1] - ax[0])
    KX, KY = np.meshgrid(center[0] + ax, center[1] + ax, indexing='ij')
    kxy = np.column_stack([KX.ravel(), KY.ravel()])
    iax = np.arange(Nk) - (Nk - 1)//2
    IIX, IIY = np.meshgrid(iax, iax, indexing='ij')
    idx = np.column_stack([IIX.ravel(), IIY.ravel()])
    eps_v, eps_c, Uv, Uc = _bands_at(kxy, params, soc)
    vbm = eps_v.max()
    eps_v -= vbm; eps_c -= vbm
    return kxy, eps_v, eps_c, Uv, Uc, float(eps_c.min()), dk_step, idx


# ---------------------------------------------------------------------------
#  Precompute Coulomb kernel building blocks
# ---------------------------------------------------------------------------
def build_kernels(kxy, eps_v, eps_c, Uv, Uc, a, r0, eps0, dk_step,
                  idx=None, wrap=True):
    """
    Precompute the (Ntot,Ntot) matrices needed by the self-consistency loop.

    Gmat  : W(k-k') <vk|vk'> <ck'|ck>            -> gap  Delta      (pairing)
    Rvv,Rvc,Rcv,Rcc : W(k-k') |<ik|mk'>|^2       -> Fock band renormalisation
    inv_norm : (dk_step)^2/(2 pi)^2 = BZ-integral weight  (1/(N a^2) on a full
               grid). W(q=0) uses the same Gygi-Baldereschi cell average, sized
               to the local plaquette spacing dk_step.

    wrap : if True, k-k' is folded into the plaquette torus (the reference
           notebook's keff/kmq modular map: q-component -> ((d+Lp//2) mod Lp)
           - Lp//2, times dk_step). This treats the plaquette as a periodic
           finite cluster, exactly as in BCS-problem-WS2-Plaquette. If False,
           the raw Euclidean k-k' is used.
    """
    if wrap and idx is not None:
        # ---- momentum transfer q = k - k' on the plaquette TORUS ------------
        # We treat the Lp x Lp plaquette as a periodic finite cluster, so the
        # momentum transfer between two k-points is taken by the minimum-image
        # convention: a difference that spans more than half the grid is folded
        # to the shorter equivalent difference on the opposite side. This is the
        # reference notebook's keff/kmq map, and it is what makes k+q and k-q
        # (needed by the kernel) close on the finite grid.
        #
        # idx[:,0], idx[:,1] are the INTEGER plaquette offsets, centred at 0
        # (so they run -h..+h for odd Lp). The raw index difference di = i - j
        # lies in [-(Lp-1), Lp-1]; the centred modular map
        #     d -> ((d + h) mod Lp) - h        (h = Lp//2)
        # folds any |d| > Lp/2 back by -/+ Lp, giving the shortest signed
        # difference in [-h, h]. Multiplying by dk_step converts index units to
        # the physical |k - k'| = sqrt(di_wrapped^2 + dj_wrapped^2) * dk_step.
        Lp = int(round(np.sqrt(len(kxy))))           # points per side
        h  = Lp//2
        di = idx[:, None, 0] - idx[None, :, 0]        # raw i-index difference
        dj = idx[:, None, 1] - idx[None, :, 1]        # raw j-index difference
        diw = ((di + h) % Lp) - h                     # minimum-image (torus) fold
        djw = ((dj + h) % Lp) - h
        q   = np.sqrt(diw.astype(float)**2 + djw.astype(float)**2) * dk_step
    else:
        # open plaquette: plain Euclidean k - k', no wrapping
        dk = kxy[:, None, :] - kxy[None, :, :]
        q  = np.linalg.norm(dk, axis=2)
    # q=0 cell average sized to the plaquette spacing (Om = area per k-point)
    Nk_eq = (2*np.pi/a)/dk_step                      # equivalent full-BZ side
    W    = _W_keldysh_mat(q, r0, eps0, a, Nk_eq**2)

    Ovv = Uv.conj() @ Uv.T          # <vk|vk'>
    Occ = Uc.conj() @ Uc.T          # <ck|ck'>
    Ovc = Uv.conj() @ Uc.T          # <vk|ck'>
    Ocv = Uc.conj() @ Uv.T          # <ck|vk'>

    Gmat = W * Ovv * Occ.conj()     # Occ.conj()[k,k'] = <ck'|ck>
    Rvv  = W * np.abs(Ovv)**2
    Rvc  = W * np.abs(Ovc)**2
    Rcv  = W * np.abs(Ocv)**2
    Rcc  = W * np.abs(Occ)**2
    inv_norm = dk_step**2/(2*np.pi)**2
    return dict(Gmat=Gmat, Rvv=Rvv, Rvc=Rvc, Rcv=Rcv, Rcc=Rcc,
                inv_norm=inv_norm, eps_v=eps_v, eps_c=eps_c, kxy=kxy, idx=idx)


def _lowest(H, vec=False):
    """Lowest eigenpair of a (dense) Hermitian matrix. Uses Lanczos (eigsh,
    fast for the extremal eigenvalue -- essential on the finer plaquette where a
    full eigh of the N^2 x N^2 kernel is the dominant one-time cost), with a
    dense-eigh fallback."""
    Hh = 0.5*(H + H.conj().T)
    try:
        from scipy.sparse.linalg import eigsh
        w, V = eigsh(Hh, k=1, which='SA')
        return (float(w[0]), V[:, 0]) if vec else float(w[0])
    except Exception:
        w, V = eigh(Hh)
        return (float(w[0]), V[:, 0]) if vec else float(w[0])


def exciton_mode(ker):
    """Lowest plaquette-BSE exciton: energy E_x and wavefunction A_k.
    H_BSE = diag(eps_c-eps_v) - inv*Gmat (direct kernel). The normal state is
    unstable to pairing exactly at mu_c-mu_v = E_x (Thouless criterion), so A_k
    is the correct finite-amplitude nucleus to seed the condensate with."""
    Ek = ker['eps_c'] - ker['eps_v']
    H  = np.diag(Ek) - ker['inv_norm']*ker['Gmat']
    return _lowest(H, vec=True)


# ---------------------------------------------------------------------------
#  Self-consistent solver for one Delta_mu
# ---------------------------------------------------------------------------
def solve_superfluid(ker, Eg, dmu, a, mix=1.0, tol=1e-9, itmax=8000,
                     seed=1e-2, init=None, seed_mode=None):
    """
    Solve Eqs. (13)-(16) self-consistently for a single input Delta_mu.

    Parameters
    ----------
    ker   : dict from build_kernels
    Eg    : equilibrium gap [eV]
    dmu   : Delta_mu = mu_c - mu_v = E_x [eV]  (control parameter)
    seed  : initial coherence amplitude |rho_vc| to break the trivial
            (Delta=0) fixed point
    init  : (rho_vc, dr_v, dr_c) warm-start from a neighbouring Delta_mu
            (continuation) -- follows the coherent BCS branch instead of
            collapsing onto the normal (Delta=0) inverted state.
    seed_mode : coherence seed profile (default = exciton eigenmode A_k)

    Returns dict with f_v, f_c, Delta, e_minus, epsr_v, epsr_c, n_Ang, n_cm2,
    inverted (bool), n_inv (max f_c), converged (bool), iters, and the state
    (rho_vc, dr_v, dr_c) for continuation.
    """
    eps_v, eps_c = ker['eps_v'], ker['eps_c']
    Gmat = ker['Gmat']; inv = ker['inv_norm']
    Rvv, Rvc, Rcv, Rcc = ker['Rvv'], ker['Rvc'], ker['Rcv'], ker['Rcc']
    Ntot = len(eps_v)
    mu_v = (Eg - dmu)/2.0
    mu_c = (Eg + dmu)/2.0
    k0 = int(np.argmin(eps_c - eps_v))

    if init is not None:
        rho_vc = init[0].copy(); dr_v = init[1].copy(); dr_c = init[2].copy()
    else:
        # seed the coherence with the exciton eigenmode A_k (the physical
        # condensate nucleus): it decays for dmu<E_x and grows for dmu>E_x,
        # so the iteration lands in the correct basin. Fall back to a band-edge
        # Gaussian if no mode supplied.
        if seed_mode is None:
            kxy = ker['kxy']
            d2 = np.sum((kxy - kxy[k0])**2, axis=1)
            seed_mode = np.exp(-d2/(2*(0.15)**2))
        rho_vc = (seed*seed_mode).astype(complex)
        dr_v = np.zeros(Ntot); dr_c = np.zeros(Ntot)

    def fixed_point(state):
        """One HF iteration: state (rho_vc, dr_v, dr_c) -> new state + f_v,f_c."""
        rho, drv, drc = state
        epsr_v = eps_v - inv*(Rvv @ drv + Rvc @ drc)     # (14) Fock (Hartree cancels)
        epsr_c = eps_c - inv*(Rcv @ drv + Rcc @ drc)
        Delta  = -inv*(Gmat @ rho)                        # (15) gap
        aa = epsr_v - mu_v; dd = epsr_c - mu_c            # (13)/(16) 2x2 at each k
        root = np.sqrt(0.25*(aa - dd)**2 + np.abs(Delta)**2)
        e_minus = 0.5*(aa + dd) - root
        phv = e_minus - dd; phc = np.conj(Delta)          # lower eigenvector
        nrm = np.sqrt(np.abs(phv)**2 + np.abs(phc)**2)
        # robust to Delta=0: the (e_minus-dd, Delta*) form degenerates to (0,0)
        # for a pure-conduction (inverted, dd<aa) normal state -> set it by hand.
        bad = nrm < 1e-12*np.maximum(np.abs(aa - dd), 1e-30)
        phv = np.where(bad, np.where(aa <= dd, 1.0, 0.0), phv)
        phc = np.where(bad, np.where(aa <= dd, 0.0, 1.0), phc)
        nrm = np.sqrt(np.abs(phv)**2 + np.abs(phc)**2) + 1e-300
        phv = phv/nrm; phc = phc/nrm
        f_v = np.abs(phv)**2; f_c = np.abs(phc)**2
        return ((phv*np.conj(phc), f_v - 1.0, f_c),
                f_v, f_c, Delta, e_minus, epsr_v, epsr_c)

    # Plain (damped) Picard substitution, as in the reference notebook. This is
    # ESSENTIAL: the condensate onset is a linear instability of the normal
    # (Delta=0) state at dmu=E_x (Thouless), and Picard *grows* that instability
    # (rate = 1+mix(L-1)>1 for the unstable mode), whereas an Anderson/DIIS
    # accelerator minimises |F(x)-x| and therefore converges ONTO the unstable
    # normal fixed point -- giving spurious decay above E_x and a metastable
    # branch below it. mix=1 reproduces the reference's direct substitution.
    state = (rho_vc, dr_v, dr_c)
    best_state, best_err = state, np.inf
    err = np.inf
    for it in range(itmax):
        new, f_v, f_c, Delta, e_minus, epsr_v, epsr_c = fixed_point(state)
        err = max(np.max(np.abs(new[0] - state[0])),
                  np.max(np.abs(new[1] - state[1])),
                  np.max(np.abs(new[2] - state[2])))
        if err < best_err:                 # keep most self-consistent iterate
            best_err, best_state = err, state
        state = ((1-mix)*state[0] + mix*new[0],
                 (1-mix)*state[1] + mix*new[1],
                 (1-mix)*state[2] + mix*new[2])
        if err < tol and it > 10:
            break
    # In the BCS regime mix=1 Picard can limit-cycle around the fixed point;
    # report the lowest-residual iterate rather than the last.
    err = best_err
    rho_vc, dr_v, dr_c = best_state
    _, f_v, f_c, Delta, e_minus, epsr_v, epsr_c = fixed_point((rho_vc, dr_v, dr_c))

    # excitation density: n = G_DEG * INT d2k/(2pi)^2 f_ck
    n_Ang = G_DEG*inv*np.sum(f_c)                    # 1/A^2 (Kramers degeneracy)
    n_cm2 = n_Ang*1e16
    return dict(dmu=dmu, f_v=f_v, f_c=f_c, Delta=Delta, e_minus=e_minus,
                epsr_v=epsr_v, epsr_c=epsr_c, kmax=k0, mu_v=mu_v, mu_c=mu_c,
                kxy=ker['kxy'], idx=ker['idx'],
                maxDelta=float(np.max(np.abs(Delta))),
                n_Ang=n_Ang, n_cm2=n_cm2, n_inv=float(f_c.max()),
                inverted=bool(f_c.max() > 0.5),
                converged=bool(err < tol), iters=it+1, err=float(err),
                state=(rho_vc, dr_v, dr_c))


# ---------------------------------------------------------------------------
#  Notebook-faithful solver (BCS-problem-WS2-Plaquette_forVitaly.nb)
# ---------------------------------------------------------------------------
def solve_notebook(ker, Eg, dmu, a, delta0=1e-4, tol=1e-6, itmax=2000):
    """One-Delta_mu 1v/1c BCS solve replicating the reference Mathematica
    notebook EXACTLY (BCS-problem-WS2-Plaquette_forVitaly.nb), i.e. with NO
    exciton (BSE A_k) seed.

    Notebook scheme (its self-consistent BCS cell):
      * cold start: Delta_k = delta0 (uniform, 1e-4), HartreeVV = HartreeCC = 0
        -- no exciton wavefunction, no warm start;
      * plain Picard (direct substitution, mix=1), up to itmax=2000 cycles;
      * each cycle: build the 2x2  [[eps_v+HVV-mu_v, Delta],[Delta*, eps_c+HCC-mu_c]]
        at every k, diagonalise it (like Mathematica `Eigensystem`), keep the
        lower-eigenvalue eigenvector (uminus=valence, vminus=conduction), then
        recompute the Fock self-energies HVV,HCC and the gap Delta from it;
      * convergence: diff = max( sum_k ||HVV_old|-|HVV|| + ||HCC_old|-|HCC|| ,
        sum_k ||Delta|-|Delta_old|| ) < tol=1e-6.

    Occupations are f_v=|uminus|^2, f_c=|vminus|^2; the Fock self-energies equal
    the notebook's HartreeVV/CC and reduce to Eq.(14) (drho_vv=f_v-1,
    drho_cc=f_c). The gap uses rho_vc = uminus*conj(vminus) (the notebook's
    Conjugate[vminus]*uminus), Eq.(15). This lands on the SAME self-consistent
    fixed point as solve_superfluid (seed-independence); it exists only to mirror
    the notebook's exact cold-start algorithm.
    """
    eps_v, eps_c = ker['eps_v'], ker['eps_c']
    Gmat = ker['Gmat']; inv = ker['inv_norm']
    Rvv, Rvc, Rcv, Rcc = ker['Rvv'], ker['Rvc'], ker['Rcv'], ker['Rcc']
    N = len(eps_v)
    mu_v = (Eg - dmu)/2.0
    mu_c = (Eg + dmu)/2.0

    Delta = np.full(N, delta0, complex)               # uniform cold seed
    HVV = np.zeros(N); HCC = np.zeros(N)              # Fock self-energies, start 0
    diff = np.inf
    for it in range(itmax):
        # (1) diagonalise the 2x2 at each k with the CURRENT bands and gap
        aa = eps_v + HVV - mu_v
        dd = eps_c + HCC - mu_c
        H2 = np.empty((N, 2, 2), complex)
        H2[:, 0, 0] = aa;            H2[:, 1, 1] = dd
        H2[:, 0, 1] = Delta;         H2[:, 1, 0] = np.conj(Delta)
        w, V = np.linalg.eigh(H2)                      # ascending; col 0 = lower
        uminus = V[:, 0, 0]                           # valence component
        vminus = V[:, 1, 0]                           # conduction component
        f_v = np.abs(uminus)**2; f_c = np.abs(vminus)**2
        # (2) store old, then recompute Fock + gap from the new eigenvectors
        HVV_old, HCC_old, Delta_old = HVV, HCC, Delta
        drv = f_v - 1.0; drc = f_c
        HVV = -inv*(Rvv @ drv + Rvc @ drc)            # Eq. (14) valence Fock
        HCC = -inv*(Rcv @ drv + Rcc @ drc)            # Eq. (14) conduction Fock
        rho_vc = uminus*np.conj(vminus)
        Delta = -inv*(Gmat @ rho_vc)                  # Eq. (15) gap
        # (3) notebook convergence test (on the moduli)
        diff = max(np.sum(np.abs(np.abs(HCC_old) - np.abs(HCC)))
                   + np.sum(np.abs(np.abs(HVV_old) - np.abs(HVV))),
                   np.sum(np.abs(np.abs(Delta) - np.abs(Delta_old))))
        if diff < tol:
            break

    n_cm2 = G_DEG*inv*np.sum(f_c)*1e16                # density (Kramers degeneracy)
    return dict(dmu=dmu, f_v=f_v, f_c=f_c, Delta=Delta, e_minus=w[:, 0],
                epsr_v=eps_v + HVV, epsr_c=eps_c + HCC, mu_v=mu_v, mu_c=mu_c,
                kxy=ker['kxy'], idx=ker['idx'],
                maxDelta=float(np.max(np.abs(Delta))), n_cm2=n_cm2,
                n_inv=float(f_c.max()), inverted=bool(f_c.max() > 0.5),
                converged=bool(diff < tol), iters=it+1, err=float(diff))


# ---------------------------------------------------------------------------
#  Driver: scan Delta_mu
# ---------------------------------------------------------------------------
def scan(Nk=41, halfwidth=0.30, n_layers=3, dmu_list=None,
         wrap=True, verbose=True):
    """Scan Delta_mu = E_x for the single-v/c (1v/1c) model.

    The condensate onset is the bare single-band exciton energy
    (E_x ~ 2.11 eV, binding ~65 meV). This underbinds MAPbI3 (Kramers doublets);
    the physical binding (~185 meV) is recovered by perovskite_superfluid_multi.py.
    The mean field stays at physical coupling (|Delta| ~ 40 meV)."""
    params = PARAMS_MAPBI3; soc = SOC_MAPBI3; a = params[-1]
    r0 = getr0(EPS_I_KELD, EPS_ENV_KELD, a*n_layers)
    eps0 = EPS0_BSE
    kxy, eps_v, eps_c, Uv, Uc, Eg, dk_step, idx = build_vc_bands(
        Nk, halfwidth, None, params, soc)
    ker = build_kernels(kxy, eps_v, eps_c, Uv, Uc, a, r0, eps0, dk_step,
                        idx=idx, wrap=wrap)
    Ex, A0 = exciton_mode(ker)                        # onset = lowest BSE eig
    if verbose:
        print(f"# n={n_layers}  plaquette {Nk}x{Nk}={len(kxy)} pts, "
              f"half-width {halfwidth} 1/A, dk={dk_step:.4f} 1/A, wrap={wrap}")
        print(f"# a={a} A  r0={r0:.3f} A  eps0={eps0}   Eg = {Eg:.4f} eV")
        print(f"# bare 1v/1c: exciton E_x = {Ex:.4f} eV "
              f"(binding {(Eg-Ex)*1e3:.0f} meV) -> condensate onset")
    if dmu_list is None:                              # fine near onset (BEC window)
        dmu_list = np.concatenate([
            np.arange(Ex-0.02, Ex+0.03, 0.0025),
            np.arange(Ex+0.03, Eg+0.09, 0.006)])
        dmu_list = np.round(dmu_list, 4)
    dmu_list = np.sort(np.asarray(dmu_list, float))   # ascending for continuation

    # Single ascending sweep, Picard-seeded with the exciton eigenmode A_k and
    # warm-started from the previous Delta_mu. Picard grows the condensate only
    # for dmu>E_x and decays it for dmu<E_x, so this gives the equilibrium
    # branch directly -- no metastable-branch bookkeeping needed.
    rows = []
    if verbose:
        print(f"# {'Dmu=Ex':>8} {'max|D|':>9} {'n(cm^-2)':>11} {'max f_c':>8} "
              f"{'inv?':>5} {'conv':>5} {'it':>6}")
    init = None
    for dmu in dmu_list:
        r = solve_superfluid(ker, Eg, float(dmu), a,
                             init=init, seed=1e-2, seed_mode=A0)
        # warm-start next dmu only from a coherent state (else re-nucleate)
        init = r['state'] if r['maxDelta'] > 1e-4 else None
        rows.append(r)
        if verbose:
            print(f"  {dmu:8.4f} {r['maxDelta']*1e3:9.4f} {r['n_cm2']:11.3e} "
                  f"{r['n_inv']:8.4f} {str(r['inverted']):>5} "
                  f"{str(r['converged']):>5} {r['iters']:6d}")
    return dict(kxy=kxy, idx=idx, eps_v=eps_v, eps_c=eps_c, Eg=Eg, Ex=Ex, a=a,
                r0=r0, eps0=eps0, Nk=Nk, halfwidth=halfwidth, dk_step=dk_step,
                n_layers=n_layers, ker=ker,
                dmu=np.array([r['dmu'] for r in rows]),
                ndens=np.array([r['n_cm2'] for r in rows]),
                maxDelta=np.array([r['maxDelta'] for r in rows]),
                finv=np.array([r['n_inv'] for r in rows]),
                rows=rows)


def scan_notebook(Nk=41, halfwidth=0.30, n_layers=3, dmu_list=None,
                  wrap=True, itmax=8000, verbose=True):
    """Delta_mu sweep using the notebook-faithful COLD-START solver
    (solve_notebook): every Delta_mu is solved independently from the uniform
    Delta=1e-4 seed -- no exciton mode, no warm-start between points. Output
    structure is identical to scan(), so make_outputs / make_arpes_figure work
    unchanged. (itmax is raised above the notebook's 2000 only so the dense
    Nk-grid fully converges; the algorithm and cold seed are the notebook's.)"""
    params = PARAMS_MAPBI3; soc = SOC_MAPBI3; a = params[-1]
    r0 = getr0(EPS_I_KELD, EPS_ENV_KELD, a*n_layers); eps0 = EPS0_BSE
    kxy, eps_v, eps_c, Uv, Uc, Eg, dk_step, idx = build_vc_bands(
        Nk, halfwidth, None, params, soc)
    ker = build_kernels(kxy, eps_v, eps_c, Uv, Uc, a, r0, eps0, dk_step,
                        idx=idx, wrap=wrap)
    Ex, _ = exciton_mode(ker)            # only to set the sweep window, NOT a seed
    if verbose:
        print(f"# COLD START (notebook)  n={n_layers}  plaquette "
              f"{Nk}x{Nk}={len(kxy)} pts, dk={dk_step:.4f} 1/A, wrap={wrap}")
        print(f"# a={a} A  r0={r0:.3f} A  eps0={eps0}   Eg = {Eg:.4f} eV")
        print(f"# uniform Delta=1e-4 seed; exciton onset E_x = {Ex:.4f} eV "
              f"(binding {(Eg-Ex)*1e3:.0f} meV)")
    if dmu_list is None:
        dmu_list = np.round(np.concatenate([
            np.arange(Ex-0.02, Ex+0.03, 0.0025),
            np.arange(Ex+0.03, Eg+0.09, 0.006)]), 4)
    dmu_list = np.sort(np.asarray(dmu_list, float))
    rows = []
    if verbose:
        print(f"# {'Dmu':>8} {'max|D|':>9} {'n(cm^-2)':>11} {'max f_c':>8} "
              f"{'inv?':>5} {'conv':>5} {'it':>6}")
    for dmu in dmu_list:
        r = solve_notebook(ker, Eg, float(dmu), a, itmax=itmax)
        rows.append(r)
        if verbose:
            print(f"  {dmu:8.4f} {r['maxDelta']*1e3:9.4f} {r['n_cm2']:11.3e} "
                  f"{r['n_inv']:8.4f} {str(r['inverted']):>5} "
                  f"{str(r['converged']):>5} {r['iters']:6d}")
    return dict(kxy=kxy, idx=idx, eps_v=eps_v, eps_c=eps_c, Eg=Eg, Ex=Ex, a=a,
                r0=r0, eps0=eps0, Nk=Nk, halfwidth=halfwidth, dk_step=dk_step,
                n_layers=n_layers, ker=ker,
                dmu=np.array([r['dmu'] for r in rows]),
                ndens=np.array([r['n_cm2'] for r in rows]),
                maxDelta=np.array([r['maxDelta'] for r in rows]),
                finv=np.array([r['n_inv'] for r in rows]),
                rows=rows)


def arpes_signal(res, eta=0.07, nw=220, wrange=None, cut='ky0'):
    """
    ARPES lesser-Green's-function signal along a k-cut through the valley
    (reference notebook's `Aless`). Two branches, each a delta function at the
    quasiparticle pole approximated by a LORENTZIAN of width eta -- exactly the
    paper's prescription (Fig. 6: "the delta functions are approximated by
    Lorentzian profiles with a finite width of 70 meV"), so eta=0.07 eV:
        conduction replica : weight f_ck at E_ck = e^-_k + mu_c
        valence            : weight f_vk at E_vk = e^-_k + mu_v = E_ck - E_x
        A(k,w) = f_ck * eta/((w-E_ck)^2+eta^2) + f_vk * eta/((w-E_vk)^2+eta^2)
    e^-_k is the lower eigenvalue of the 2x2 built with the RENORMALISED bands
    eps^r (paper Eq. 4 / notebook EpoleS = e^-_k + mu_c in the VBM=0 reference).

    Returns dict(kx, w, A[nw,Ncut], E_ck, E_vk, fc, fv).
    """
    kxy = res['kxy']; idx = res['idx']
    if cut == 'ky0':
        sel = np.where(idx[:, 1] == 0)[0]            # central row (ky = k_valley)
    else:
        sel = np.where(idx[:, 0] == 0)[0]
    sel = sel[np.argsort(idx[sel, 0] if cut == 'ky0' else idx[sel, 1])]
    kaxis = kxy[sel, 0] if cut == 'ky0' else kxy[sel, 1]
    kx = kaxis - kaxis.mean()
    em = np.real(res['e_minus'])[sel]
    E_ck = em + res['mu_c']
    E_vk = em + res['mu_v']
    fc = res['f_c'][sel]; fv = res['f_v'][sel]
    if wrange is None:
        lo = min(E_vk.min(), E_ck.min()) - 0.35
        hi = max(E_vk.max(), E_ck.max()) + 0.35
    else:
        lo, hi = wrange
    w = np.linspace(lo, hi, nw)
    A = (fc[None, :]*eta/((w[:, None]-E_ck[None, :])**2 + eta**2)
         + fv[None, :]*eta/((w[:, None]-E_vk[None, :])**2 + eta**2))
    return dict(kx=kx, w=w, A=A, E_ck=E_ck, E_vk=E_vk, fc=fc, fv=fv)


# ---------------------------------------------------------------------------
#  Plain-text (.txt) exports of all data
# ---------------------------------------------------------------------------
def save_txt(out, outdir, densities=ARPES_DENSITIES):
    """Write every dataset that also goes into the .npz/.pkl as human-readable
    .txt files in `outdir`:

      superfluid_scan.txt          Delta_mu sweep summary (one row per Delta_mu):
                                   dmu, n, max|Delta|, max f_c, inverted flag.
      superfluid_kresolved.txt     full self-consistent solution at every k for
                                   every Delta_mu: dmu, kx, ky, f_v, f_c, |Delta|,
                                   e_minus, eps^r_v, eps^r_c.
      superfluid_arpes.txt         ARPES bands/weights per k for all density
                                   panels: target_n, n, dmu, kx, E_ck, E_vk, fc, fv.
      superfluid_arpes_n{d}.txt    one file per panel: the 2D A^<(k,w) map in
                                   long format (kx, omega, A).
    """
    Eg, Ex = out['Eg'], out['Ex']
    meta = (f"MAPbI3 n={out.get('n_layers', 3)} superfluid scan | "
            f"Eg={Eg:.6f} eV  Ex={Ex:.6f} eV  binding={(Eg-Ex)*1e3:.1f} meV | "
            f"density n = G_DEG * INT d2k/(2pi)^2 f_c (Kramers degeneracy G_DEG={G_DEG:.0f})")

    # (1) scan summary -------------------------------------------------------
    dmu, nd, mD, finv = out['dmu'], out['ndens'], out['maxDelta'], out['finv']
    f1 = os.path.join(outdir, 'superfluid_scan.txt')
    np.savetxt(f1, np.column_stack([dmu, nd, mD*1e3, finv,
                                    (finv > 0.5).astype(float)]),
               header=meta + "\ndmu_eV    n_cm2    maxDelta_meV    max_f_c    "
                             "inverted(1/0)",
               fmt=['%.6f', '%.6e', '%.6e', '%.6f', '%.0f'])
    print("saved", f1)

    # (2) k-resolved solution for every Delta_mu -----------------------------
    rows = out['rows']
    blocks = []
    for r in rows:
        m = len(r['f_c'])
        blocks.append(np.column_stack([
            np.full(m, r['dmu']), r['kxy'][:, 0], r['kxy'][:, 1],
            np.real(r['f_v']), np.real(r['f_c']), np.abs(r['Delta']),
            np.real(r['e_minus']), np.real(r['epsr_v']), np.real(r['epsr_c'])]))
    f2 = os.path.join(outdir, 'superfluid_kresolved.txt')
    np.savetxt(f2, np.vstack(blocks),
               header=meta + "\ndmu_eV  kx_invA  ky_invA  f_v  f_c  absDelta_eV  "
                             "e_minus_eV  epsr_v_eV  epsr_c_eV",
               fmt='%.6e')
    print("saved", f2, f"({len(rows)} dmu x {len(rows[0]['f_c'])} k pts)")

    # (3) ARPES bands/weights for all density panels (combined file) ---------
    picks = [min(rows, key=lambda r: abs(r['n_cm2'] - nt)) for nt in densities]
    ars   = [arpes_signal(r) for r in picks]
    lo = min(min(a['E_vk'].min(), a['E_ck'].min()) for a in ars) - 0.35
    hi = max(max(a['E_vk'].max(), a['E_ck'].max()) for a in ars) + 0.35
    blocks = []
    for nt, r, ar in zip(densities, picks, ars):
        m = len(ar['kx'])
        blocks.append(np.column_stack([
            np.full(m, nt), np.full(m, r['n_cm2']), np.full(m, r['dmu']),
            ar['kx'], ar['E_ck'], ar['E_vk'], ar['fc'], ar['fv']]))
    f3 = os.path.join(outdir, 'superfluid_arpes.txt')
    np.savetxt(f3, np.vstack(blocks),
               header=meta + "\ntarget_n_cm2  n_cm2  dmu_eV  kx_invA  E_ck_eV  "
                             "E_vk_eV  f_c  f_v",
               fmt='%.6e')
    print("saved", f3)

    # (4) per-panel 2D A^<(k,w) maps (one file each) -------------------------
    for nt, r in zip(densities, picks):
        ar = arpes_signal(r, wrange=(lo, hi))          # common w-axis (as plotted)
        KX, W = np.meshgrid(ar['kx'], ar['w'])         # A has shape (nw, nkx)
        cols = np.column_stack([KX.ravel(), W.ravel(), ar['A'].ravel()])
        fp = os.path.join(outdir, f'superfluid_arpes_n{_n_label(nt)}.txt')
        np.savetxt(fp, cols,
                   header=meta + f"\ntarget_n={nt:.3e} cm^-2  n={r['n_cm2']:.6e} "
                                 f"cm^-2  dmu={r['dmu']:.6f} eV  "
                                 f"{'inverted' if r['inverted'] else 'BEC'}"
                                 "\nkx_invA  omega_eV  A_kw",
                   fmt='%.6e')
        print("saved", fp)


def _pick_bec_bcs(coh):
    """Representative BEC and BCS solutions from the coherent rows.
    BEC = a developed but NON-inverted point (max f_c in (0.05,0.5]); if none,
    the lowest-density coherent point. BCS = the highest-density point."""
    bec_candidates = [r for r in coh if 0.05 < r['n_inv'] <= 0.5]
    rbec = (max(bec_candidates, key=lambda r: r['n_inv']) if bec_candidates
            else min(coh, key=lambda r: r['n_cm2']))
    rbcs = max(coh, key=lambda r: r['n_cm2'])
    return rbec, rbcs


def _pick_three(coh, n_mid=5e12):
    """BEC, an intermediate point nearest n_mid, and BCS (highest density)."""
    rbec, rbcs = _pick_bec_bcs(coh)
    rmid = min(coh, key=lambda r: abs(r['n_cm2'] - n_mid))
    return rbec, rmid, rbcs


def make_arpes_figure(out, fname="superfluid_arpes.png",
                      densities=ARPES_DENSITIES, ncols=3):
    """ARPES A^<(k,w) maps at a set of fixed excitation densities (default
    ARPES_DENSITIES). For each target the nearest scan row is used; a common
    energy axis is shared across panels, laid out in a grid of `ncols` columns."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out['rows']
    if not rows:
        print("no solution -> skip ARPES"); return
    picks = [min(rows, key=lambda r: abs(r['n_cm2'] - nt)) for nt in densities]
    ars = [arpes_signal(r) for r in picks]
    lo = min(min(a['E_vk'].min(), a['E_ck'].min()) for a in ars) - 0.35
    hi = max(max(a['E_vk'].max(), a['E_ck'].max()) for a in ars) + 0.35
    n = len(picks)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.4*nrows),
                             squeeze=False)
    axflat = axes.ravel()
    for ax, r, nt in zip(axflat, picks, densities):
        ar = arpes_signal(r, wrange=(lo, hi))          # common w-axis
        pc = ax.pcolormesh(ar['kx'], ar['w'], ar['A'], shading='auto',
                           cmap='turbo')
        ax.plot(ar['kx'], ar['E_ck'], 'w--', lw=0.7, alpha=0.7)
        ax.plot(ar['kx'], ar['E_vk'], 'w:', lw=0.7, alpha=0.7)
        ax.set_xlabel(r'$k_x - k_{\rm valley}$ (1/$\AA$)')
        ax.set_ylabel(r'$\omega$ (eV)')
        ax.set_title(f"target $n$={nt:.0e} cm$^{{-2}}$\n"
                     f"$n$={r['n_cm2']:.2e} cm$^{{-2}}$  "
                     f"$\\Delta\\mu$={r['dmu']:.3f} eV  "
                     f"{'inverted' if r['inverted'] else 'BEC'}")
        fig.colorbar(pc, ax=ax, fraction=0.046, label=r'$A^<(k,\omega)$')
    for ax in axflat[n:]:                              # hide unused cells
        ax.axis('off')
    fig.tight_layout(); fig.savefig(fname, dpi=140)
    print("saved", fname, "at n =", [f"{r['n_cm2']:.2e}" for r in picks])


def make_outputs(out, scan_png="superfluid_scan.png",
                 tex="superfluid_results.tex"):
    """Figure (scan_png) + LaTeX results table (tex)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dmu, nd, mD = out['dmu'], out['ndens'], out['maxDelta']
    finv = out['finv']; rows = out['rows']; kxy = out['kxy']
    # population-inversion onset
    inv_mask = finv > 0.5
    n_c = nd[inv_mask].min() if inv_mask.any() else np.nan

    fig = plt.figure(figsize=(14.5, 4.2))
    # (1) n and max|Delta| vs Delta_mu
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.semilogy(dmu, np.maximum(nd, 1e2), 'o-', ms=3, color='#1f77b4')
    ax1.set_xlabel(r'$\Delta\mu = E_x$ (eV)'); ax1.set_ylabel(r'$n$ (cm$^{-2}$)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    if np.isfinite(n_c):
        ax1.axhline(n_c, ls='--', c='gray', lw=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(dmu, mD*1e3, 's-', ms=3, color='#d62728')
    ax1b.set_ylabel(r'$\max_k|\Delta_k|$ (meV)', color='#d62728')
    ax1b.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_title('density & gap vs $\\Delta\\mu$')

    # pick a BEC-side and a BCS-side converged point for the f_c maps
    def fc_map(ax, r, ttl):
        Nk = int(np.sqrt(len(kxy)))
        KX = kxy[:, 0].reshape(Nk, Nk); KY = kxy[:, 1].reshape(Nk, Nk)
        pc = ax.pcolormesh(KX-KX.mean(), KY-KY.mean(),
                           r['f_c'].reshape(Nk, Nk), shading='auto',
                           cmap='inferno', vmin=0, vmax=1)
        ax.set_title(ttl); ax.set_xlabel(r'$k_x-k_M$'); ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, fraction=0.046, label=r'$f^{\rm sf}_{ck}$')
    # BEC / intermediate (n~5e12) / BCS
    coh = [r for r in rows if r['maxDelta'] > 5e-3]
    if coh:
        rbec, rmid, rbcs = _pick_three(coh)
        fc_map(fig.add_subplot(1, 4, 2), rbec,
               f"BEC-side  $\\Delta\\mu$={rbec['dmu']:.3f}\n$n$={rbec['n_cm2']:.1e}")
        fc_map(fig.add_subplot(1, 4, 3), rmid,
               f"intermediate  $\\Delta\\mu$={rmid['dmu']:.3f}\n$n$={rmid['n_cm2']:.1e}")
        fc_map(fig.add_subplot(1, 4, 4), rbcs,
               f"BCS-side  $\\Delta\\mu$={rbcs['dmu']:.3f}\n$n$={rbcs['n_cm2']:.1e}")
    fig.tight_layout(); fig.savefig(scan_png, dpi=140)
    print("saved", scan_png)

    # LaTeX results table (every other row to keep it compact)
    with open(tex, "w") as f:
        f.write("Equilibrium gap $E_g=%.4f$~eV; exciton $E_{\\rm x}=%.4f$~eV "
                "(binding %.0f~meV, bare 1v/1c); population-inversion onset "
                "$n_c\\approx%.2e~\\text{cm}^{-2}$.\\\\[4pt]\n" %
                (out['Eg'], out['Ex'], (out['Eg']-out['Ex'])*1e3, n_c))
        f.write("\\begin{center}\\small\n\\begin{tabular}{rrrrc}\n\\toprule\n")
        f.write("$\\Delta\\mu=E_x$ (eV) & $\\max_k|\\Delta_k|$ (meV) & "
                "$n$ (cm$^{-2}$) & $\\max_k f^{\\rm sf}_{ck}$ & inverted \\\\\n\\midrule\n")
        for r in rows[::2]:
            f.write("%.4f & %.3f & %.3e & %.4f & %s \\\\\n" %
                    (r['dmu'], r['maxDelta']*1e3, r['n_cm2'], r['n_inv'],
                     "yes" if r['inverted'] else "no"))
        f.write("\\bottomrule\n\\end{tabular}\\end{center}\n")
    print("saved", tex)


if __name__ == "__main__":
    import os, sys, pickle
    # Usage:
    #   python perovskite_superfluid_clean.py [Nk] [halfwidth]        # A_k-seeded scan
    #   python perovskite_superfluid_clean.py [Nk] [halfwidth] cold   # notebook cold start
    # All outputs are written to ./out_n{n_layers}_superfluid/.
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    hw = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    cold = (len(sys.argv) > 3 and sys.argv[3].lower().startswith("cold"))
    n_layers = 3
    pre = "superfluid_coldstart" if cold else "superfluid"

    outdir = f"out_n{n_layers}_superfluid"
    os.makedirs(outdir, exist_ok=True)
    path = lambda name: os.path.join(outdir, name)

    out = (scan_notebook(Nk=Nk, halfwidth=hw, n_layers=n_layers) if cold
           else scan(Nk=Nk, halfwidth=hw, n_layers=n_layers))

    np.savez(path(f"{pre}_scan.npz"),
             dmu=out['dmu'], ndens=out['ndens'], maxDelta=out['maxDelta'],
             finv=out['finv'], Eg=out['Eg'], Ex=out['Ex'])
    print("saved", path(f"{pre}_scan.npz"))

    make_outputs(out, scan_png=path(f"{pre}_scan.png"),
                 tex=path(f"{pre}_results.tex"))
    make_arpes_figure(out, fname=path(f"{pre}_arpes.png"))
    save_txt(out, outdir)
    try:
        with open(path(f"{pre}_rows.pkl"), "wb") as fh:
            pickle.dump({k: out[k] for k in ('rows', 'kxy', 'idx', 'Eg', 'Ex',
                         'Nk', 'halfwidth', 'dk_step')}, fh)
        print("saved", path(f"{pre}_rows.pkl"))
    except Exception as e:
        print("pickle skipped:", e)
    print("READY  ->", outdir)
