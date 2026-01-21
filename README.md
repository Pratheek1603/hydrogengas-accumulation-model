# Hydrogen Gas Accumulation Model

A Python implementation of a **1-D coupled gas accumulation model** for **H₂, CH₄, N₂, and He**, describing time-dependent gas buildup in subsurface systems subject to deep flux inputs, diffusive and advective leakage, and chemical or microbial gas transformation.

This repository provides **numerical and analytic solutions** to the governing mass-balance equations and reproduces all figures associated with the model analysis.

---

## 📌 Model Overview

The model simulates the accumulation of multiple gases in a 1-D control volume through time, accounting for:

- Constant **deep gas fluxes**
- **Advective leakage** from the reservoir
- **Diffusive loss** of individual gas species
- **Chemical or microbial consumption** of H₂
- **Coupled H₂ → CH₄ conversion** (e.g., Sabatier-type processes)

The governing equations form a system of **linear ordinary differential equations (ODEs)** with constant coefficients, allowing both **numerical integration** and **closed-form analytic solutions**.

---

## ⚙️ Governing Equations

For each gas species \( Q_i \) (He, N₂, H₂, CH₄), the model solves:

- Source terms from deep fluxes  
- Loss terms from advection and diffusion  
- Reaction terms coupling H₂ consumption and CH₄ generation  

The coupled system is implemented explicitly in the code and corresponds to the governing equations described in the associated manuscript.

---

## 🧮 Numerical & Analytic Solution

- **Numerical integration** is performed using `scipy.integrate.solve_ivp`
- **Analytic closed-form solutions** are implemented for validation and steady-state analysis
- Numerical and analytic solutions are directly compared to verify correctness

The model uses **days** as the internal time unit; results are commonly visualized in **years**.

---

## 📊 Outputs & Figures

Running the main script reproduces the following outputs:

- Time evolution of absolute gas amounts (linear and log scale)
- Normalized gas composition versus time (mol %)
- Numerical vs analytic solution comparison
- Sensitivity of steady-state CH₄/H₂ ratio to conversion rate (γ)
- Ternary trajectory (N₂–H₂–CH₄)
- Simple compositional inversion to estimate system age for a target gas mixture

All figures are generated automatically using `matplotlib` and `python-ternary`.

---

## 🚀 How to Run

### Requirements
- Python ≥ 3.8  
- NumPy  
- SciPy  
- Matplotlib  
- python-ternary  

Install dependencies:
```bash
pip install numpy scipy matplotlib python-ternary
