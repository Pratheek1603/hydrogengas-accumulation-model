import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# USER INPUT: PHILIPPINE SITE DATA
#

# Observed gas composition at Zambales Ophiolite Complex seeps (%)
obs_H2  = 58.0   # measured H2 %
obs_N2  = 2.0    # measured N2 %
obs_CH4 = 39.0   # measured CH4 %
obs_He  = 1.0    # measured He %

# Depth (m)
depth = 100   # shallow ultramafic exposures

# Temperature gradient (°C/km)
Tgrad = 35    # typical for Zambales ophiolite

# Caprock thickness
L = 30        # meters (assume serpentinized clay/seal)

# MODEL PARAMETERS (adjusted for Zambales)

# Fluxes (assumed ongoing serpentinization fluxes)
Phi_He = 1e-7
Phi_N2 = 1e-6
Phi_H2 = 1e-4
Phi_CH4 = 5e-5

# Leakage / escape (open seep system)
beta_adv = 5e-5   # advective leakage (d^-1)

# Diffusive losses (scaled, minor)
beta_diff_He  = 1e-7
beta_diff_N2  = 1e-6
beta_diff_H2  = 1e-4
beta_diff_CH4 = 5e-5

# Chemical alteration (minor for open seep)
alpha_H2  = 1e-6       # H2 oxidation (d^-1)
alpha_CH4 = 1e-7       # CH4 biodegradation (d^-1)

# H2 → CH4 Sabatier reaction (low for open system)
gamma_H2 = 5e-6

# Governing Equations (Analytical)


def QHe(t):
    return (Phi_He - beta_diff_He)/beta_adv * (1 - np.exp(-beta_adv*t))

def QN2(t):
    return (Phi_N2 - beta_diff_N2)/beta_adv * (1 - np.exp(-beta_adv*t))

def QH2(t):
    A = (Phi_H2 - beta_diff_H2)
    lam = alpha_H2 + beta_adv + gamma_H2
    return A/lam * (1 - np.exp(-lam*t))

def QCH4(t):
    A = (Phi_H2 - beta_diff_H2)
    lam1 = alpha_H2 + beta_adv + gamma_H2
    lam2 = alpha_CH4 + beta_adv
    return (gamma_H2 * A/lam1) * ((1-np.exp(-lam2*t))/lam2 + (np.exp(-lam2*t)-np.exp(-lam1*t))/(alpha_CH4 - alpha_H2 - gamma_H2))

def gas_fractions(t):
    """Returns normalized mol fractions (%) of each gas."""
    h2  = QH2(t)
    n2  = QN2(t)
    ch4 = QCH4(t)
    he  = QHe(t)

    total = h2 + n2 + ch4 + he
    return 100*h2/total, 100*n2/total, 100*ch4/total, 100*he/total

# Age Estimation Function


def misfit(t):
    model = gas_fractions(t[0])
    obs   = np.array([obs_H2, obs_N2, obs_CH4, obs_He])
    return np.sum((np.array(model) - obs)**2)

result = minimize(misfit, x0=[1000], bounds=[(1, 1e9)])
best_age = result.x[0]

print("---------------------------------------------------")
print(f"Estimated age of the hydrogen system: {best_age:.1f} days")
print(f"= {best_age/365:.2f} years")
print(f"= {best_age/(365*1000):.3f} kyr")
print("---------------------------------------------------")

# Plot gas fraction evolution

times = np.logspace(0, 9, 500)  # days

H2_f, N2_f, CH4_f, He_f = [], [], [], []
for t in times:
    h2, n2, ch4, he = gas_fractions(t)
    H2_f.append(h2)
    N2_f.append(n2)
    CH4_f.append(ch4)
    He_f.append(he)

plt.figure(figsize=(8,6))
plt.semilogx(times/365, H2_f,  label="H₂")
plt.semilogx(times/365, N2_f,  label="N₂")
plt.semilogx(times/365, CH4_f, label="CH₄")
plt.semilogx(times/365, He_f,  label="He")

plt.axvline(best_age/365, color='k', linestyle='--', label="Estimated age")
plt.xlabel("Time (years, log scale)")
plt.ylabel("Gas Fraction (%)")
plt.title("Zambales Ophiolite - Gas Composition Evolution Over Time")
plt.legend()
plt.grid(True)
plt.show()
