##### This script calculates the unbinned-cross-section for a given qT-array.
##### It stores the results in a pickel file. The resulting array is linked to the parameters ser (Q,c1,c2,c3), so the results can be called by choosing the values of these parameters.

import lhapdf
import numpy as np
import pickle
from scipy import integrate
import os
import time
import itertools
import multiprocessing as mp
from mcfit import Hankel # type: ignore
import numpy as np
import time
from scipy.interpolate import interp1d


# -------------------------------------------------------
# Sets the x-data-arrays
# -------------------------------------------------------

x1_data = np.array([0.5, 1.5, 2.5])

x2_data = np.array([0.5, 1.5, 2.5, 3.5])

x3_data = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

# -------------------------------------------------------
# Worker initializer: runs ONCE per worker process
# -------------------------------------------------------

def init_worker(pdf_name="MSHT20lo_as130"):
    global p
    print(f"[Init] Loading LHAPDF '{pdf_name}' in PID {os.getpid()}")
    p = lhapdf.getPDFSet(pdf_name).mkPDF(0)
    # Now every worker has a global PDF object "p"

def As(Q):
    return p.alphasQ(Q)
def a_s(Q):
    return As(Q)/(4*np.pi)

# -------------------------------------------------------
# Constants and functions
# -------------------------------------------------------

# sets the common constants
CA = 3
CF = 4/3
TF = 1/2
Tr = 1/2
gamma_euler = 0.5772156649015328
b0 = 2 * np.exp(-gamma_euler)
zeta2 = 1.64493
zeta3 = 1.20206
zeta4 = 1.08232
zeta5 = 1.03693
mc = 1.50
mb = 4.75

# sets the bT-prescription
n = 10
def b_prime(bt,Q):
    return (bt**n + (b0/Q)**n )**(1/n)
def b_star(bt,Q,bmax):
    return b_prime(bt,Q) / (1 + (b_prime(bt,Q)/bmax)**n )**(1/n)

# sets the bmax value
bmax = 1

# sets the flavour-scheme
nf = 4

# sets the gluon fraction momenta
s_sqrt = 13000
def x1(y,Q):
    return Q * np.exp(y) / s_sqrt
def x2(y,Q):
    return Q * np.exp(-y) / s_sqrt


# sets the values of beta function
# the expressions are taken from [1701.01404]
beta0 = lambda nf: 11 * CA / 3 - 4 * TF * nf / 3 # 1-loop
beta1 = lambda nf: 34 * (CA**2)/3 - 20*CA * TF*nf/3 - 4 * CF*TF*nf # 2-loop
beta2 = lambda nf: 2857 * (CA**3) / 54 + (2 * CF**2 - 205*CF*CA/9 - 1415*CA**2/27) * TF * nf + (44*CF/9 + 158*CA/27) * (TF**2) * (nf**2) #3-loop
# 4-loop and 5-loop are missing in this code. they are in the previous reference

# sets the values of cusp anomalous dimension for a gluon
Gammacusp0_g = lambda nf: 4 * CA # 1-loop
Gammacusp1_g = lambda nf: 4 * CA * ( (67/9 - 2*zeta2) * CA - 10*nf/9) # 2-loop
Gammacusp2_g =  lambda nf: 4 * CA * ( (CA ** 2) * ( 245/6 - 269*zeta2/9 + 22*zeta3/3 + 22*zeta4 ) + CA * nf * (-209/27 + 40*zeta2/9 - 28*zeta3/3) + CF * nf * (-55/6 + 8 * zeta3) - 4 * (nf**2)/27 ) # 3-loop
# 4-loop and 5-loop are missing in this code. they are in [2001.11377] and [1812.11818] formula (13), respectively

# sets the values of vector FF anomalous dimension
# it is defined with factor 2 (standard for TMD physics)
# 3-loop expresion is taken from [1004.3653]
gammaV1_g = lambda nf: - 22*CA/3 + 4*nf/3 # 1-loop
gammaV2_g = lambda nf: 2*((CA**2)*(-692/27 +11*zeta2/3+2*zeta3) +CA*nf*(128/27 -2*zeta2/3) +2*CF*nf) # 2-loop
gammaV3_g = lambda nf: 2*((CA**3)*(-97186/729 +6109*zeta2/81 -319*zeta4/3 +122*zeta3/3 -40*zeta2*zeta3/3 -16*zeta5)+(CA**2)*nf*(30715/1458 -1198*zeta2/81 +82*zeta4/3 +356*zeta3/27)
            +CF*CA*nf*(1217/27 -2*zeta2 -8*zeta4 -152*zeta3/9)+CA*(nf**2)*(-269/1458 +20*zeta2/27 -56*zeta3/27)-(CF**2)*nf -11*CF*(nf**2)/9) # 3-loop
# 4-loop is missing in this code. it can be taken from [2202.04660]

# sets the values of the RAD
d10_g = lambda nf: 0.0 # 1-loop
d11_g = lambda nf: Gammacusp0_g(nf) / 2

d20_g = lambda nf: (CA**2) * (404/27 - 14*zeta3) - CA*nf*56/27  # 2-loop
d21_g = lambda nf: Gammacusp1_g(nf) / 2
d22_g = lambda nf: Gammacusp0_g(nf) * beta0(nf) / 4

# sets the perturbative Sudakov factor for one gluon (LL, NLL, N2LL)
LT = lambda bt,mu: np.log((mu**2) * (bt**2) / (b0**2))

def SA_f1g_nnll(bt,mu,zeta,mub,zeta0):
    def integrand(mup):
        return (1/mup) * ( ( a_s(mup) * Gammacusp0_g(nf) + (a_s(mup)**2) * Gammacusp1_g(nf) + (a_s(mup))**3 * Gammacusp2_g(nf)) * np.log(zeta/mup**2) + a_s(mup) * gammaV1_g(nf) + (a_s(mup)**2) * gammaV2_g(nf) )
    SA_nnll_integral = integrate.quad( lambda mup: integrand(mup), mub, mu, epsrel = 1e-3, limit = 100 )[0]
    return ( (d10_g(nf) + d11_g(nf) * LT(bt,mub)) * a_s(mub) + ( d20_g(nf) + d21_g(nf) * LT(bt,mub) + d22_g(nf) * (LT(bt,mub))**2 ) * (a_s(mub))**2 ) * np.log(zeta/zeta0) + SA_nnll_integral

def SA_f1g_nll(bt,mu,zeta,mub,zeta0):
    def integrand(mup):
        return (1/mup) * ( ( a_s(mup) * Gammacusp0_g(nf) + (a_s(mup)**2) * Gammacusp1_g(nf)) * np.log(zeta/mup**2) + a_s(mup) * gammaV1_g(nf) )
    SA_nll_integral = integrate.quad( lambda mup: integrand(mup), mub, mu, epsrel = 1e-3, limit = 100 )[0]
    return ( (d10_g(nf) + d11_g(nf) * LT(bt,mub)) * a_s(mub) ) * np.log(zeta/zeta0) + SA_nll_integral

def SA_f1g_ll(bt,mu,zeta,mub,zeta0):
    def integrand(mup):
        return (1/mup) * ( ( a_s(mup) * Gammacusp0_g(nf) ) * np.log(zeta/mup**2)  )
    SA_ll_integral = integrate.quad( lambda mup: integrand(mup), mub, mu, epsrel = 1e-3, limit = 100 )[0]
    return SA_ll_integral

def Conv_sumCgf1g_NLO(x, bt, mui, zetai):

    # Cache constants
    as_over_2pi = As(mui) / (2 * np.pi)
    lt = LT(bt, mui)
    CA2 = 2 * CA
    pdf_x = p.xfxQ(0, x, mui) / x

    """
    g -> g delta term
    """
    def Cgg_delta():
        return pdf_x * (1 + as_over_2pi *
                        CA * (-0.5 * lt**2 + lt * np.log(mui**2 / zetai) - np.pi**2 / 12))

    """
    g -> g plus distribution term
    """
    def Cgg_plus():
        pref = as_over_2pi * (-lt) * CA2
        pdf0 = pdf_x

        def integrand(xh):
            xp = x / xh
            return pref * (1 / (1 - xh)) * (p.xfxQ(0, xp, mui) / xp - pdf0)

        I = integrate.quad(integrand, x, 1.0, epsrel=1e-4, limit=500)[0]

        return I + pref * pdf0 * np.log(1 - x)

    """
    g -> g finite term
    """
    def Cgg_finite():
        pref = as_over_2pi * (-lt) * CA2

        def integrand(xh):
            xp = x / xh
            pdf = p.xfxQ(0, xp, mui) / xp
            kernel = (1 - xh) / xh + xh * (1 - xh)
            return pref * pdf * kernel / xh

        return integrate.quad(integrand, x, 1.0, epsrel=1e-4, limit=500)[0]

    """
    g -> q term
    """
    flavors = np.concatenate((np.arange(-4, 0), np.arange(1, 5)))  # q + q̄

    def Cgq():
        pref1 = as_over_2pi * (-lt) * CF
        pref2 = as_over_2pi * CF

        def integrand(xh):
            xp = x / xh
            sum_q = np.sum(p.xfxQ(flavors, xp, mui) )/xp

            term1 = pref1 * (1 + (1 - xh)**2) / xh
            term2 = pref2 * xh
            return (sum_q / xh) * (term1 + term2)

        return integrate.quad(integrand, x, 1.0, epsrel=1e-4, limit=500)[0]

    """
    final result
    """
    return Cgg_delta() + Cgg_plus() + Cgg_finite() + Cgq()

# -------------------------------------------------------
# Definition of the convolution integrand
# -------------------------------------------------------


def integrand_W_nnll(bt, y, Q, c1, c2, c3):
    """
    c1 -> SV in SA, c2 -> SV in Hard Scale, c3 -> SV in TMD
    """

    # --- constants / scaled quantities ---
    Q_sv = c2 * Q
    h = 1e-4
    # normalization factor that appears repeatedly: 1 / (2^(2/n) * bmax)
    norm_inv = (1 / (2 ** (2 / n) * bmax))
    eps = 1e-300  # tiny safeguard against exact zero denominators

    conv = Conv_sumCgf1g_NLO

    # --- compute b_star values needed for finite differences around bmax ---
    bt_plus  = b_star(bmax + h, Q_sv, bmax)
    bt_minus = b_star(bmax - h, Q_sv, bmax)
    bt_max   = b_star(bmax, Q_sv, bmax)

    # --- SA derivative (g_A) using central difference ---
    mub_plus_SA  = (c1 * b0) / bt_plus
    mub_minus_SA = (c1 * b0) / bt_minus
    zeta_plus_SA  = mub_plus_SA ** 2
    zeta_minus_SA = mub_minus_SA ** 2

    # original expression: np.abs((2*SA(bt_plus,...)-2*SA(bt_minus,...)) / (2*h)) /(2**(2/n)*bmax)
    # simplifies to: np.abs((SA_plus - SA_minus) / h) * norm_inv
    SA_plus  = SA_f1g_nnll(bt_plus,  Q_sv, Q_sv**2, mub_plus_SA,  zeta_plus_SA)
    SA_minus = SA_f1g_nnll(bt_minus, Q_sv, Q_sv**2, mub_minus_SA, zeta_minus_SA)
    dSA_db = (2*SA_plus - 2*SA_minus) / (2*h)
    g_A = np.abs(dSA_db) * norm_inv

    # --- g_f for the two gluons: central differences of Conv_sumCgf1g_NLO normalized by value at bt_max ---
    # compute TMD scales for plus/minus and max
    mub_plus_TMD  = (c3 * b0) / bt_plus
    mub_minus_TMD = (c3 * b0) / bt_minus
    zeta_plus_TMD  = mub_plus_TMD ** 2
    zeta_minus_TMD = mub_minus_TMD ** 2

    mub_max = (c3 * b0) / bt_max
    zeta_max = mub_max ** 2

    # x1, x2
    x1_val = x1(y, Q_sv)
    x2_val = x2(y, Q_sv)

    # Evaluate conv at required points (reuse local variables)
    # Note: conv = Conv_sumCgf1g_NLO or cached wrapper
    f1_plus  = conv(x1_val, bt_plus, mub_plus_TMD,  zeta_plus_TMD)
    f1_minus = conv(x1_val, bt_minus, mub_minus_TMD, zeta_minus_TMD)
    f1_max   = conv(x1_val, bt_max, mub_max, zeta_max)

    f2_plus  = conv(x2_val, bt_plus,  mub_plus_TMD,  zeta_plus_TMD)
    f2_minus = conv(x2_val, bt_minus, mub_minus_TMD, zeta_minus_TMD)
    f2_max   = conv(x2_val, bt_max,   mub_max,        zeta_max)

    # avoid exact zero denominators (very unlikely but safe)
    denom1 = f1_max if abs(f1_max) > eps else eps
    denom2 = f2_max if abs(f2_max) > eps else eps

    # original: np.abs((f_plus - f_minus) / (2*h * f(bt_max))) * norm_inv
    g_f_1 = np.abs((f1_plus - f1_minus) / (2 * h * denom1)) * norm_inv
    g_f_2 = np.abs((f2_plus - f2_minus) / (2 * h * denom2)) * norm_inv

    # --- NP model with continuity constraint (SNP) ---
    g_total = g_A + g_f_1 + g_f_2
    # b_dagger^2 = ((bt^n + bmax^n)^(2/n) - bmax^2)
    b_dagger2 = ( (bt ** n + bmax ** n) ** (2.0 / n) ) - (bmax ** 2)
    SNP_bt = np.exp(- g_total * b_dagger2)

    # --- final b_star and scales for final evaluation ---
    b_star_value = b_star(bt, Q_sv, bmax)
    mu_bstar_sv_SA  = (c1 * b0) / b_star_value
    mu_bstar_sv_TMD = (c3 * b0) / b_star_value
    zeta_bstar_TMD  = mu_bstar_sv_TMD ** 2

    F1 = conv(x1_val, b_star_value, mu_bstar_sv_TMD, zeta_bstar_TMD)
    F2 = conv(x2_val, b_star_value, mu_bstar_sv_TMD, zeta_bstar_TMD)

    SA_bstar = SA_f1g_nnll(b_star_value, Q_sv, Q_sv**2, mu_bstar_sv_SA, mu_bstar_sv_SA ** 2)

    return F1 * F2 * SNP_bt * np.exp(-2 * SA_bstar)


# -------------------------------------------------------
# Integatrion on y for LHCb kinematics
# -------------------------------------------------------


def integrand_W_nnll_integrated_y_hankel(bt,Q,c1,c2,c3):

    ymax = 4.5
    ymin = 2
    bin_width = ymax - ymin

    y_int = integrate.quad(lambda y: integrand_W_nnll(bt,y,Q,c1,c2,c3), ymin, ymax, epsrel = 1e-3, limit = 50)[0]

    return (1/(2*np.pi)) * y_int / bin_width

# -------------------------------------------------------
# Definition of the cross-section as qT * C[f1g f1g Sh]
# The NLO anomalous dimensions of the TMDShF is zero
# -------------------------------------------------------

def binned_func(qt_array, Q, c1, c2, c3):

    # 1. b-grid
    blow = 1e-10
    bhigh = 20
    N = 500
    b_grid = np.logspace(np.log10(blow), np.log10(bhigh), N)

    # 2. Compute W(b)
    Wb = np.array([
        integrand_W_nnll_integrated_y_hankel(b, Q, c1, c2, c3) for b in b_grid
    ])
    Wb = np.nan_to_num(Wb, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Hankel transform (API for mcfit 0.0.22)
    hankel = Hankel(b_grid, 0)

    # Call object with ONLY W(b)
    q_out, F_q = hankel(Wb)      # <-- CORRECT for mcfit 0.0.22

    # 4. q * F(q)
    f_q = q_out * F_q
    f_q = np.nan_to_num(f_q)

    # 5. Interpolate to qt_array
    f_interp = interp1d(q_out, f_q, kind = 'cubic', bounds_error=False, fill_value=0.0)

    # 6. Integration over the kt-bin
    y_th = np.array([ integrate.quad( lambda kt: float(f_interp(kt)), x_value - 0.5, x_value + 0.5, epsrel = 1e-3, limit = 100 )[0] for x_value in qt_array ])
    norm = np.sum(y_th)

    return y_th / norm


# -------------------------------------------------------
# Definition of the function which I want to compute.
# -------------------------------------------------------
def compute_one(params):

    start = time.time()

    Q, c1, c2, c3 = params
    print(f"Computing for Q={Q}, c1={c1}, c2={c2}, c3={c3}")

    x_max_map = {6.6: x1_data, 7.9: x2_data, 11: x3_data}
    
    x = x_max_map[Q]
    y = binned_func(x, Q, c1, c2, c3)

    end = time.time()

    print(y)
    print(end - start)

    return (Q, c1, c2, c3, y)


# --------------------------------------------------
# Main program that runs everything in parallel in macOS
# --------------------------------------------------

def main():
    Q_values = [6.6, 7.9, 11]
    c_values = [0.5, 1, 2]

    start = time.time()

    # Build all parameter tuples
    params_list = list(itertools.product(Q_values, c_values, c_values, c_values))

    # Use all available CPU cores
    with mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
        results = pool.map(compute_one, params_list)

    # Store results as a dictionary for later notebook access
    result_dict = {}
    for Q, c1, c2, c3, y in results:
        result_dict[(Q, c1, c2, c3)] = y

    # Save with pickle
    with open("binned_cross_section_NNLL.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    end = time.time()

    print(f"Execution time: {end - start:.4f} seconds")
    print("All results computed and saved to binned_cross_section_NNLL.pkl")


if __name__ == "__main__":
    main()