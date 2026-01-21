"""
hydro_gas_model.py
Full reproduction of the 1D gas accumulation model (He, N2, H2, CH4) from the paper.

Outputs produced:
 - Time evolution of absolute gas amounts (log scale and linear)
 - Normalized composition vs time (mol %)
 - Analytic steady-state / closed-form solutions comparison
 - CH4/H2 steady-state sensitivity vs gamma
 - Ternary plot (N2-H2-CH4) trajectory
 - Simple inversion: estimate age that best matches a target composition (e.g., Bourakebougou)

Units:
 - All rates and time internal to model: days (as in Table 1)
 - Plots show time converted to years for readability.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import ternary
import math

# -------------------------
# PARAMETERS (from Table 1 in the paper)
# Units: days (d, d^-1) and m^3/m^2.d for fluxes as in the paper
# -------------------------
params = {
    # advective leakage coefficient (d^-1)
    "beta_adv": 2.74e-11,            # (half-time ~100 My in paper)
    # deep fluxes (m3/m2.d) from Table 1
    "Phi_He": 4.76e-11,
    "Phi_N2": 1.00e-09,
    "Phi_H2": 1.00e-03,
    "Phi_CH4": 0.0,                  # none specified as input

    # diffusive leakage (m.d^-1) (product solubility * D / pressure approximations)
    "beta_diff_He": 1.00e-11,
    "beta_diff_N2": 9.70e-11,
    "beta_diff_H2": 2.96e-10,
    "beta_diff_CH4": 2.80e-12,

    # chemical/bacterial alteration rates (d^-1)
    "alpha_H2": 5.00e-02,            # H2 -> H+ (0.05 d^-1)
    "alpha_CH4": 1.00e-08,           # CH4 alteration (very small)
    "gamma_H2": 1.00e-07             # H2 -> CH4 (Sabatier); can vary for sensitivity
}

# Convenience: local variables for quick reference
beta_adv = params['beta_adv']
Phi_He = params['Phi_He']
Phi_N2 = params['Phi_N2']
Phi_H2 = params['Phi_H2']
Phi_CH4 = params['Phi_CH4']
beta_diff_He = params['beta_diff_He']
beta_diff_N2 = params['beta_diff_N2']
beta_diff_H2 = params['beta_diff_H2']
beta_diff_CH4 = params['beta_diff_CH4']
alpha_H2 = params['alpha_H2']
alpha_CH4 = params['alpha_CH4']
gamma_H2 = params['gamma_H2']


# -------------------------
# ODE system (Q in m^3 per m^2 area or equivalent volumetric unit)
# Equations from the paper (Eqs. 5-7)
# dQ_He/dt  = (Phi_He - beta_diff_He) - beta_adv * Q_He
# dQ_N2/dt  = (Phi_N2 - beta_diff_N2) - beta_adv * Q_N2
# dQ_H2/dt  = (Phi_H2 - beta_diff_H2) - (alpha_H2 + beta_adv + gamma_H2) * Q_H2
# dQ_CH4/dt = gamma_H2 * Q_H2 - (alpha_CH4 + beta_adv) * Q_CH4 - beta_diff_CH4
# -------------------------
def gas_ode(t, Q, p):
    QHe, QN2, QH2, QCH4 = Q
    dQHe = (p["Phi_He"] - p["beta_diff_He"]) - p["beta_adv"] * QHe
    dQN2 = (p["Phi_N2"] - p["beta_diff_N2"]) - p["beta_adv"] * QN2
    dQH2 = (p["Phi_H2"] - p["beta_diff_H2"]) - (p["alpha_H2"] + p["beta_adv"] + p["gamma_H2"]) * QH2
    dQCH4 = p["gamma_H2"] * QH2 - (p["alpha_CH4"] + p["beta_adv"]) * QCH4 - p["beta_diff_CH4"]
    return [dQHe, dQN2, dQH2, dQCH4]


# -------------------------
# Closed-form analytic solutions (paper eqns 8-11)
# Valid because the forcing terms are constants and the ODEs are linear with constant coefficients
# -------------------------
def analytic_QHe(t, p):
    # QHe(t) = (PhiHe - beta_diffHe)/beta_adv * (1 - exp(-beta_adv * t))
    denom = p["beta_adv"]
    num = (p["Phi_He"] - p["beta_diff_He"])
    # handle near-zero beta_adv safely
    if denom <= 0:
        return num * t
    return (num / denom) * (1.0 - np.exp(-denom * t))

def analytic_QN2(t, p):
    denom = p["beta_adv"]
    num = (p["Phi_N2"] - p["beta_diff_N2"])
    if denom <= 0:
        return num * t
    return (num / denom) * (1.0 - np.exp(-denom * t))

def analytic_QH2(t, p):
    k = p["alpha_H2"] + p["beta_adv"] + p["gamma_H2"]
    num = (p["Phi_H2"] - p["beta_diff_H2"])
    if k <= 0:
        return num * t
    return (num / k) * (1.0 - np.exp(-k * t))

def analytic_QCH4(t, p):
    # Using equation (11) from the paper (rearranged). We'll evaluate safely.
    # To avoid numerical instability, we construct expression carefully.
    aH2 = p["alpha_H2"]
    aCH4 = p["alpha_CH4"]
    b = p["beta_adv"]
    g = p["gamma_H2"]
    Ph = p["Phi_H2"]
    bdH2 = p["beta_diff_H2"]
    # Precompute k's
    kH2 = aH2 + b + g
    kCH4 = aCH4 + b
    numH2 = Ph - bdH2
    if kH2 == 0 or kCH4 == 0:
        # fallback numerical integration or approximate
        return 0.0
    term1 = (g * numH2) / kH2
    # bracketed term from paper: [ (1 - exp(-kCH4*t))/kCH4 + (exp(-kCH4*t)-exp(-kH2*t))/(kCH4 - kH2) ]
    # handle kCH4 ~= kH2
    eps = 1e-30
    if abs(kCH4 - kH2) < 1e-15:
        # limit where kCH4 -> kH2, use l'Hospital style series expansion
        # approximate second term via t*exp(-kCH4*t)
        t = np.array(t)
        bracket = (1.0 - np.exp(-kCH4 * t)) / kCH4 + t * np.exp(-kCH4 * t)
    else:
        t = np.array(t)
        bracket = (1.0 - np.exp(-kCH4 * t)) / kCH4 + (np.exp(-kCH4 * t) - np.exp(-kH2 * t)) / (kCH4 - kH2)
    return term1 * bracket

# -------------------------
# RUN NUMERICAL SOLUTION
# -------------------------
# choose time vector in days but plot in years
t_end_days = 1.5e8   # simulate up to 150 million days (~410k years) or larger as in paper (they used up to 150 Myr but their units were days - paper timescales are huge)
# The paper's text mentions up to 150 million years; their rates are in d^-1. To avoid confusion,
# choose a long enough t_end; user may adjust. Here set to 5e7 days ≈ 137k years; but original paper shows up to 150 million years (huge).
# To mimic paper's long times we use 5e8 days ~ 1.37 million years; adjust as needed.
t_end_days = 1.5e8   # keep as 150,000,000 days (~410k years) - consistent with paper large runs
n_points = 2500
t_eval = np.linspace(0.0, t_end_days, n_points)

Q0 = [0.0, 0.0, 0.0, 0.0]  # start empty accumulation
sol = solve_ivp(fun=lambda t, Q: gas_ode(t, Q, params),
                t_span=(0.0, t_end_days), y0=Q0, t_eval=t_eval,
                rtol=1e-9, atol=1e-12, method='RK45')

QHe = sol.y[0]
QN2 = sol.y[1]
QH2 = sol.y[2]
QCH4 = sol.y[3]

# Also compute analytic forms for comparison:
QHe_analytic = analytic_QHe(t_eval, params)
QN2_analytic = analytic_QN2(t_eval, params)
QH2_analytic = analytic_QH2(t_eval, params)
# QCH4 analytic is more complex; use numerical integration or ℚCH4 analytic approximate:
# We'll compute analytic QCH4 by numerically evaluating the formula using the functions above:
QCH4_analytic = analytic_QCH4(t_eval, params)

# -------------------------
# PLOT 1: Absolute amounts vs time (log scale)
# -------------------------
plt.figure(figsize=(10,6))
years = t_eval / 365.25  # convert days to years for readability

plt.loglog(years + 1e-12, np.maximum(QHe, 1e-30), label='He (num)', linewidth=2)
plt.loglog(years + 1e-12, np.maximum(QN2, 1e-30), label='N2 (num)', linewidth=2)
plt.loglog(years + 1e-12, np.maximum(QH2, 1e-30), label='H2 (num)', linewidth=2)
plt.loglog(years + 1e-12, np.maximum(QCH4, 1e-30), label='CH4 (num)', linewidth=2)

# overlay analytic if helpful
plt.loglog(years + 1e-12, np.maximum(QHe_analytic, 1e-30), '--', linewidth=1, label='He (analytic)')
plt.loglog(years + 1e-12, np.maximum(QN2_analytic, 1e-30), '--', linewidth=1, label='N2 (analytic)')
plt.loglog(years + 1e-12, np.maximum(QH2_analytic, 1e-30), '--', linewidth=1, label='H2 (analytic)')

plt.xlabel('Time (years)')
plt.ylabel('Absolute gas amount Q (m^3 / m^2 or relative volume unit)')
plt.title('Fig. 6-like: Absolute amounts (log-log) vs time')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()

# -------------------------
# Compute normalized compositions (mol% approx from volumes)
# Normalize across the four species at each time point
# -------------------------
Qtotal = QHe + QN2 + QH2 + QCH4
# Avoid divide-by-zero
Qtotal_safe = Qtotal.copy()
Qtotal_safe[Qtotal_safe == 0] = 1.0
XHe = QHe / Qtotal_safe
XN2 = QN2 / Qtotal_safe
XH2 = QH2 / Qtotal_safe
XCH4 = QCH4 / Qtotal_safe

# -------------------------
# PLOT 2: Composition vs time (linear)
# -------------------------
plt.figure(figsize=(10,6))
plt.plot(years, XH2*100, label='H2 (mol%)', linewidth=2)
plt.plot(years, XN2*100, label='N2 (mol%)', linewidth=2)
plt.plot(years, XCH4*100, label='CH4 (mol%)', linewidth=2)
plt.plot(years, XHe*100, label='He (mol%)', linewidth=2)
plt.xlabel('Time (years)')
plt.ylabel('Gas composition (mol%)')
plt.title('Fig. 7-like: Relative composition vs time (linear)')
plt.legend()
plt.grid(True)
plt.xscale('linear')
plt.tight_layout()

# -------------------------
# PLOT 3: Zoom / match to Bourakebougou (example)
# Bourakebougou observed: H2 ~ 98 %, N2 1 %, CH4 1 % and He 500 ppm (0.05%)
# We try to find model time where composition matches close to these %s
# -------------------------
target = {"H2": 0.98, "N2": 0.01, "CH4": 0.01, "He": 0.0005}  # mole fractions
# compute simple L2 distance between model composition and target
comp_matrix = np.vstack([XH2, XN2, XCH4, XHe]).T

dist = np.linalg.norm(comp_matrix - np.array([target["H2"], target["N2"], target["CH4"], target["He"]]), axis=1)
idx_best = np.argmin(dist)
best_time_years = years[idx_best]
best_comp = comp_matrix[idx_best]

print(f"\nBest-match model time: {best_time_years:.2f} years (index {idx_best})")
print(f"Model composition at that time: H2={best_comp[0]*100:.3f}%, N2={best_comp[1]*100:.3f}%, CH4={best_comp[2]*100:.3f}%, He={best_comp[3]*100:.6f}%")
print(f"Target composition (Bourakébougou): H2={target['H2']*100:.3f}%, N2={target['N2']*100:.3f}%, CH4={target['CH4']*100:.3f}%, He={target['He']*100:.6f}%")

# Zoomed plot around best match
window_span = int(max(10, n_points // 100))  # small window for zoom
i0 = max(0, idx_best - window_span)
i1 = min(n_points - 1, idx_best + window_span)

plt.figure(figsize=(8,5))
plt.plot(years[i0:i1], XH2[i0:i1]*100, label='H2', linewidth=2)
plt.plot(years[i0:i1], XN2[i0:i1]*100, label='N2', linewidth=2)
plt.plot(years[i0:i1], XCH4[i0:i1]*100, label='CH4', linewidth=2)
plt.plot(years[i0:i1], XHe[i0:i1]*100, label='He', linewidth=2)
plt.axvline(best_time_years, color='k', ls='--', label=f'best match ~ {best_time_years:.1f} yr')
plt.xlabel('Time (years)')
plt.ylabel('Mol %')
plt.title('Zoom around best match to Bourakébougou-like composition')
plt.legend()
plt.grid(True)
plt.tight_layout()

# -------------------------
# PLOT 4: CH4/H2 steady-state sensitivity vs gamma (γ)
# Use analytic steady-state expressions
# For steady-state (large t), QH2_ss = (Phi_H2 - beta_diff_H2)/(alpha_H2 + beta_adv + gamma)
# QCH4_ss ~ gamma * QH2_ss / (alpha_CH4 + beta_adv)   [ignoring small diffusive term]
# So CH4/H2 ratio steady-state = ( gamma / (alpha_CH4 + beta_adv) ) * ( (Phi_H2 - beta_diff_H2) / (alpha_H2 + beta_adv + gamma) ) / QH2_ss
# simplifies to paper form: (gamma/(alpha_CH4 + beta_adv)) * ((alpha_H2 + beta_adv)/(alpha_H2 + beta_adv + gamma))
gammas = np.logspace(-9, -4, 300)  # vary gamma across orders of magnitude including paper values
ratio_list = []
for g in gammas:
    numerator = g / (alpha_CH4 + beta_adv)
    fraction = (alpha_H2 + beta_adv) / (alpha_H2 + beta_adv + g)
    ratio = numerator * fraction
    ratio_list.append(ratio)

plt.figure(figsize=(8,5))
plt.loglog(gammas, ratio_list, linewidth=2)
plt.xlabel('γ (H2→CH4 conversion rate) [d^-1]')
plt.ylabel('Steady-state CH4/H2 ratio (mol/mol)')
plt.title('Sensitivity: steady-state CH4/H2 vs γ')
plt.grid(True, which='both', ls='--')
plt.tight_layout()

# -------------------------
# PLOT 5: Ternary (N2 - H2 - CH4) using python-ternary
# We'll plot the trajectory in the ternary triangle (N2, H2, CH4), ignoring He (low %)
# Normalize points to N2 + H2 + CH4
# -------------------------
tri_points = []
for i in range(len(t_eval)):
    a = QN2[i]
    b = QH2[i]
    c = QCH4[i]
    s = a + b + c
    if s <= 0:
        tri_points.append((0.0,0.0,0.0))
    else:
        # ternary library expects tuple (left,right,top) or in our usage (N2,H2,CH4)
        tri_points.append((a/s, b/s, c/s))

# Create ternary figure
scale = 1.0
figure, tax = ternary.figure(scale=scale)
tax.boundary(linewidth=1.0)
tax.gridlines(multiple=0.1, color="gray")
tax.set_title("Ternary trajectory: N2 (left) - H2 (right) - CH4 (top)")
# Convert to points scaled to 'scale' for plotting
pts_scaled = [(p[0]*scale, p[1]*scale, p[2]*scale) for p in tri_points if (p[0]+p[1]+p[2])>0]
# plot sparse subset for performance (e.g., every 10th point)
tax.plot(pts_scaled[::10], linewidth=1.0, marker='o', markersize=2)
# highlight best match point
best_tri = tri_points[idx_best]
tax.scatter([(best_tri[0]*scale, best_tri[1]*scale, best_tri[2]*scale)], marker='s', color='red', label='best match')
tax.legend()
tax.ticks(axis='lbr', multiple=0.2, linewidth=1)
tax.clear_matplotlib_ticks()
plt.tight_layout()

# -------------------------
# OPTIONAL: Inversion via scalar minimization to find time that minimizes composition distance
# (A more rigorous approach would do a multi-parameter inversion; here we just find t)
# We'll find t in [0, t_end_days] minimizing L2 distance between model composition and target
# Using interpolation on the computed solution for speed
# -------------------------
from scipy.interpolate import interp1d

# Interpolants for each normalized component as function of time (days)
interp_H2 = interp1d(t_eval, XH2, kind='cubic', bounds_error=False, fill_value=(XH2[0], XH2[-1]))
interp_N2 = interp1d(t_eval, XN2, kind='cubic', bounds_error=False, fill_value=(XN2[0], XN2[-1]))
interp_CH4 = interp1d(t_eval, XCH4, kind='cubic', bounds_error=False, fill_value=(XCH4[0], XCH4[-1]))
interp_He = interp1d(t_eval, XHe, kind='cubic', bounds_error=False, fill_value=(XHe[0], XHe[-1]))

def comp_dist_days(t_day, target):
    # distance between model composition at t_day and target (L2 norm)
    h2 = float(interp_H2(t_day))
    n2 = float(interp_N2(t_day))
    ch4 = float(interp_CH4(t_day))
    he = float(interp_He(t_day))
    vec = np.array([h2, n2, ch4, he])
    targ = np.array([target['H2'], target['N2'], target['CH4'], target['He']])
    return np.linalg.norm(vec - targ)

# minimize dist in days (bounded)
res = minimize_scalar(lambda tt: comp_dist_days(tt, target),
                      bounds=(0.0, t_end_days), method='bounded', options={'xatol':1e-6})
t_found_days = res.x
t_found_years = t_found_days / 365.25
print(f"\nInversion (minimization) found best-fit time: {t_found_days:.2f} days ≈ {t_found_years:.2f} years")
print(f"Distance at best-fit: {res.fun:.6e}")

# show composition at that time
h2_fit = float(interp_H2(t_found_days))
n2_fit = float(interp_N2(t_found_days))
ch4_fit = float(interp_CH4(t_found_days))
he_fit = float(interp_He(t_found_days))
print(f"Composition at best-fit: H2={h2_fit*100:.3f}%, N2={n2_fit*100:.3f}%, CH4={ch4_fit*100:.3f}%, He={he_fit*100:.6f}%")

# -------------------------
# Show all plots
# -------------------------
plt.show()

