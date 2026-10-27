"""Leakage-aware Clifford-twirl RB approximation, not compiled sequence RB.

Uses the shared channel's computational survival, CZ error contraction and
seepage. Leakage population has its own dynamics and changes the decay floor.
The leakage reservoir is isotropically mixed between gates (an approximation).
The fit can fail its residual check; it is not guaranteed to return the hidden F.
"""
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from chipcal.sim.oracle import IRBResult, STANDARD_IRB, _single_qubit_error, _spam_visibility
from chipcal.sim.multimode import gate_fidelity


def _branch_channels(theta, config):
    """[(weight, contraction, transfer, survival, frame factor per Clifford)] over the noise branches.

    The frame factor is the idle Z error a calibrated-frame offset leaves over each Clifford's single-qubit layers
    (chipcal.sim.realism); the CZ's own frame error is already inside its channel. 1 without realism."""
    from chipcal.sim.multimode import noise_branches
    from chipcal.sim.realism import CLIFFORD_1Q_LAYERS, SQ_LAYER_S, frame_rb_contraction
    out = []
    for weight, branch in noise_branches(theta):
        fidelity, diag = gate_fidelity(config, branch)
        survival = diag["computational_survival"]
        seepage = diag["seepage"]
        contraction = np.clip((4*fidelity-survival)/3, 0., survival)
        transfer = np.array([[survival,seepage],[1-survival,1-seepage]])
        frame = (frame_rb_contraction(branch.frame_offset_c, branch.frame_offset_t, CLIFFORD_1Q_LAYERS*SQ_LAYER_S)
                 if getattr(branch, "calibrated_frames", False) else 1.)
        out.append((weight, contraction, transfer, survival, frame))
    return out


def run_irb(theta, config, rng, protocol=STANDARD_IRB):
    channels = _branch_channels(theta, config)
    survival = float(sum(w*s for w, _, _, s, _ in channels))
    lengths = np.asarray(protocol.lengths)
    sq = np.clip(1-4/3*8.25*_single_qubit_error(theta),0,1)
    visibility = _spam_visibility(theta)
    counts = np.zeros((2,protocol.n_seeds,len(lengths)),int)
    leaked = np.zeros_like(counts)
    for mode in (0,1):
        for seed in range(protocol.n_seeds):
            # Actual integer gate-count variation, mean 1.5 CZ/Clifford in ref.
            native = np.r_[0,np.cumsum(rng.integers(1,3,size=int(max(lengths))))]
            for j,m in enumerate(lengths):
                gates = int(native[m]+mode*m)
                # Each shot runs in one charge-parity branch: the survival curves mix.
                comp = p0 = 0.
                for weight, contraction, transfer, _, frame in channels:
                    branch_comp = (np.linalg.matrix_power(transfer,gates) @ [1.,0.])[0]
                    contrast = contraction**gates * sq**m * frame**m
                    comp += weight*branch_comp
                    p0 += weight*(.25*branch_comp + .75*visibility*contrast)
                # Separate outcomes make survival and leakage shots consistent.
                probs = np.clip([p0,comp-p0,1-comp],0,1)
                draw = rng.multinomial(protocol.shots,probs/np.sum(probs))
                counts[mode,seed,j], leaked[mode,seed,j] = draw[0],draw[2]
    # Independent uniform-computational-input one-gate leakage probe.
    probe = rng.binomial(protocol.shots,survival,size=protocol.n_seeds)
    survival_hat = probe.mean()/protocol.shots
    survival_var = max(survival_hat*(1-survival_hat),1/protocol.shots)/(protocol.n_seeds*protocol.shots)
    def model(m,a,p,b):
        return a*p**m+b

    def fit(seed_counts):
        """(decay p, its std error, reduced chi^2) of one interleave mode's seed-averaged survival."""
        y = seed_counts.mean(0)/protocol.shots
        sigma = np.maximum(seed_counts.std(0,ddof=1)/protocol.shots/np.sqrt(len(seed_counts)),
                           np.sqrt(np.maximum(y*(1-y),1/protocol.shots)/(len(seed_counts)*protocol.shots)))
        try:
            popt,pcov=curve_fit(model,lengths,y,p0=[.7,.97,.2],sigma=sigma,absolute_sigma=True,
                               bounds=([0,0,0],[1,1,1]),maxfev=10000)
            return (float(popt[1]), float(np.sqrt(max(pcov[1,1],0))),
                    float(np.sum(((y-model(lengths,*popt))/sigma)**2)/max(1,len(lengths)-3)))
        except (RuntimeError,ValueError):
            return 0., 1., 1e12

    (pref, err_ref, chi_ref), (pint, err_int, chi_int) = fit(counts[0]), fit(counts[1])
    reduced = [chi_ref, chi_int]
    ratio = pint/max(pref,1e-9)
    variance = (.75/max(pref,1e-9))**2*(err_int**2+ratio**2*err_ref**2)+survival_var/16
    # Seed bootstrap: the analytic propagation ignores seed-to-seed scatter of the decay, so take the
    # larger of the two (an LCB must not exceed the true fidelity much more often than nominal).
    boot = []
    for _ in range(64):
        pick = rng.integers(0, protocol.n_seeds, size=protocol.n_seeds)
        b_ref, b_int = fit(counts[0][pick])[0], fit(counts[1][pick])[0]
        if b_ref > 0:
            boot.append((3*b_int/b_ref + survival_hat)/4)
    if len(boot) > 8:
        variance = max(variance, float(np.var(boot, ddof=1)))
    # A single exponential does not describe leakage-dressed decays exactly; with many shots the fit's
    # reduced chi^2 then exceeds 1 even for a well-determined decay. Widen sigma by the Birge ratio rather
    # than calling the fit unidentifiable.
    sigma = float(np.sqrt(variance*max(1.,*reduced)))
    fhat = float(np.clip((3*ratio+survival_hat)/4,0,1))
    # Below p_ref ~ 0.5 the reference decays into the floor within a few Cliffords and the ratio is not
    # identifiable.
    reliable = pref > .5 and np.isfinite(sigma) and sigma < .1
    lcb = float(np.clip(fhat-2*sigma,0,1)) if reliable else 0.
    ds=xr.Dataset({"y_survival": (("interleave_modes","seeds","number_of_cliffords"),counts),
                   "y_leakage": (("interleave_modes","seeds","number_of_cliffords"),leaked),
                   "computational_probe_counts": (("seeds",),probe)},
                  coords={"interleave_modes":[0,1],"seeds":np.arange(protocol.n_seeds),"number_of_cliffords":lengths},
                  attrs={"node":"cz_rb","shots":protocol.shots,"simulation_model":"multimode_channel_twirl_rb_v1",
                         "reference_cz_per_clifford":1.5,"fit_reduced_chi_squared":reduced,
                         "fit_reliable":bool(reliable),"floor_model":"free fit; generated with leakage/seepage dynamics",
                         "limitation":"Clifford twirl with mixed leakage reservoir; not compiled coherent sequence simulation",
                         "independent_leakage_probe_shots":int(protocol.n_seeds*protocol.shots)})
    return IRBResult(fhat,sigma,lcb,pref,pint,ds)
