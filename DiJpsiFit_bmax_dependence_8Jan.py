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
import scipy.special as special


# -------------------------------------------------------
# Worker initializer: runs ONCE per worker process
# -------------------------------------------------------

def init_worker(pdf_name="MSHT20lo_as130"):
    global p
    print(f"[Init] Loading LHAPDF '{pdf_name}' in PID {os.getpid()}")
    p = lhapdf.getPDFSet(pdf_name).mkPDF(0)
    # Now every worker has a global PDF object "p"

x_lower_limit = 1e-6 
mu_lower_limit = 1

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
mc = 1.40
mb = 4.75


# sets the variable flavor-scheme
def nf_variable(mu):
    mu = np.asarray(mu)
    conditions = [
        mu <= mc,
        (mu > mc) & (mu <= mb),
        mu > mb
    ]
    choices = [3, 4, 5]
    return np.select(conditions, choices)

# sets the bT-prescription
n = 10
def b_prime(bt,Q):
    return (bt**n + (b0/Q)**n )**(1/n)
def b_star(bt,Q,bmax):
    return b_prime(bt,Q) / (1 + (b_prime(bt,Q)/bmax)**n )**(1/n)


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

# sets the explicitily RAD resummed expression up to alpha_s^3
# From 1803.11089
def d_function(bt, mu, order):
    nf = nf_variable(mu)
    X = beta0(nf) * a_s(mu) * LT(bt,mu)
    if order == 0:
        return ( - Gammacusp0_g(nf) / (2 * beta0(nf)) ) * np.log(1 - X)
    elif order == 1:
        return (1 / (2 * beta0(nf) * (1-X))) * ( - (beta1(nf) * Gammacusp0_g(nf) / beta0(nf)) * (np.log(1-X) + X) + Gammacusp1_g(nf) * X )
    elif order == 2:
        return (1/(1-X)**2) * ( (Gammacusp0_g(nf) * beta1(nf)**2 / (4 * beta0(nf)**3)) * (np.log(1-X)**2 - X**2) 
                               + (beta1(nf) * Gammacusp1_g(nf)/(4 * beta0(nf)**2)) * (X**2 - 2 * X - 2 * np.log(1-X)) + (Gammacusp0_g(nf) * beta2(nf)/(4*beta0(nf)**2))*(X**2) 
                               - (Gammacusp2_g(nf)/(4 * beta0(nf))) * X * (X-2) + d20_g(nf) )

# sets the sudakov factor at NLL and NNLL
# it is related with the evolution kernel as Log[R] = - SA
# these definitions are for only one gluon, a factor of 2 should be added in the cross-section calculation
# the "improved" refers to the version with the resummed RAD
def SA_f1g_nnll_improved(bt,mu,zeta,mub,zeta0):
    def integrand(mup):
        nf = nf_variable(mup)
        return (1/mup) * ( ( a_s(mup) * Gammacusp0_g(nf) + (a_s(mup)**2) * Gammacusp1_g(nf) + (a_s(mup)**3) * Gammacusp2_g(nf)) * np.log(zeta/mup**2) + a_s(mup) * gammaV1_g(nf) + (a_s(mup)**2) * gammaV2_g(nf) )
    SA_nnll_integral = integrate.quad( lambda mup: integrand(mup), mub, mu, epsrel = 1e-3, limit = 100 )[0]
    return ( d_function(bt, mub, 0) + a_s(mub) * d_function(bt,mub,1) + (a_s(mub)**2) * d_function(bt,mub,2) ) * np.log(zeta/zeta0) + SA_nnll_integral

def SA_f1g_nll_improved(bt,mu,zeta,mub,zeta0):
    def integrand(mup):
        nf = nf_variable(mup)
        return (1/mup) * ( ( a_s(mup) * Gammacusp0_g(nf) + (a_s(mup)**2) * Gammacusp1_g(nf) ) * np.log(zeta/mup**2) + a_s(mup) * gammaV1_g(nf) )
    SA_nll_integral = integrate.quad( lambda mup: integrand(mup), mub, mu, epsrel = 1e-3, limit = 100 )[0]
    return ( d_function(bt, mub, 0) + a_s(mub) * d_function(bt,mub,1)  ) * np.log(zeta/zeta0) + SA_nll_integral

# sets the TMDPDF at small_b, i.e., Convolution of Wilson coeffifient up to NLO and PDF

LT = lambda bt,mu: np.log((mu**2) * (bt**2) / (b0**2))

# NLO
# 1502.05354 (Higgs production)
def Conv_sumCgf1g_NLO(x, bt, mui, zetai):

    nf = nf_variable(mui)

    # Cache constants
    as_over_2pi = As(mui) / (2 * np.pi)
    lt = LT(bt, mui)
    CA2 = 2 * CA
    pdf_x = p.xfxQ(0, x, mui) / x

    # ------------------------------------------------------------------
    # g → g delta term
    # ------------------------------------------------------------------
    def Cgg_delta():
        return pdf_x * (1 + as_over_2pi *
                        CA * (-0.5 * lt**2 + lt * np.log(mui**2 / zetai) - np.pi**2 / 12))

    # ------------------------------------------------------------------
    # g → g plus distribution term
    # ------------------------------------------------------------------
    def Cgg_plus():
        pref = as_over_2pi * (-lt) * CA2
        pdf0 = pdf_x

        def integrand(xh):
            xp = x / xh
            return pref * (1 / (1 - xh)) * (p.xfxQ(0, xp, mui) / xp - pdf0)

        I = integrate.quad(integrand, x, 1.0, epsrel=1e-4, limit=500)[0]

        return I + pref * pdf0 * np.log(1 - x)

    # ------------------------------------------------------------------
    # g → g finite term
    # ------------------------------------------------------------------
    def Cgg_finite():
        pref = as_over_2pi * (-lt) * CA2

        def integrand(xh):
            xp = x / xh
            pdf = p.xfxQ(0, xp, mui) / xp
            kernel = (1 - xh) / xh + xh * (1 - xh)
            return pref * pdf * kernel / xh

        return integrate.quad(integrand, x, 1.0, epsrel=1e-4, limit=500)[0]

    # ------------------------------------------------------------------
    # g → q term
    # ------------------------------------------------------------------
    flavors = np.concatenate((np.arange(-nf, 0), np.arange(1, nf+1)))  # q + q̄

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

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    return Cgg_delta() + Cgg_plus() + Cgg_finite() + Cgq()

# LO
def Conv_sumCgf1g_LO(x,bt,mui,zetai): # keep the same argument structure as for the NLO definitions
    return (p.xfxQ(0,x,mui)/x)


# -------------------------------------------------------
# Definition of the convolution integrand
# -------------------------------------------------------

# sets the function setting a specific constraint on the evaluation of some value
def safe_eval(value, constraint):
    return value if constraint else 0

# - - - - - - - Calculation of the integrand of the Convolution with uncertainties bands - - - - - - - #
"""
    Scale variation: mu_i --> c1 * (b0/b_star),  zeta_i --> c2**2 * (b0/b_star)**2
    NP model: We impose do not get oscillations by the n-framework
    The definition of the NP model and the n-framework can be obtained from the overleaf
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - #
def integrand_W_nll(bt, y, Q, c1, c2, bmax):

    # --- constants / scaled quantities --- #
    Q_sv = Q # here we can vary the hard scale as well
    h = 1e-4

    # normalization factor that appears repeatedly in the n-framework: 1 / (2^(2/n) * bmax)
    norm_inv = (1 / (2 ** (2 / n) * bmax))
    eps = 1e-300  # tiny safeguard against exact zero denominators

    # x1, x2
    x1_val = x1(y, Q_sv)
    x2_val = x2(y, Q_sv)

    # more simple notation for the TMDPDF at small-b
    conv = Conv_sumCgf1g_LO

    # --- compute b_star values needed for finite differences around bmax ---#
    bt_plus  = b_star(bmax + h, Q_sv, bmax)
    bt_minus = b_star(bmax - h, Q_sv, bmax)
    bt_max   = b_star(bmax, Q_sv, bmax)

    # --- Scale variation definition --- #
    mub_plus  = (c1 * b0) / bt_plus
    mub_minus = (c1 * b0) / bt_minus
    mub_max = (c1 * b0) / bt_max
    zeta_plus  = c2**2 * (b0/bt_plus)**2 
    zeta_minus = c2**2 * (b0/bt_minus)**2
    zeta_max = c2**2 * (b0/bt_max)**2

    # --- SA calculation with the n-framework NP model in the full-range of b_T --- #
    #SA_plus = safe_eval( SA_f1g_nll_improved(bt_plus,  Q_sv, Q_sv**2, mub_plus,  zeta_plus) , (1 < mub_plus))
    #SA_minus = safe_eval( SA_f1g_nll_improved(bt_minus, Q_sv, Q_sv**2, mub_minus, zeta_minus) , (1 < mub_minus))

    SA_plus = SA_f1g_nll_improved(bt_plus,  Q_sv, Q_sv**2, mub_plus,  zeta_plus) 
    SA_minus = SA_f1g_nll_improved(bt_minus, Q_sv, Q_sv**2, mub_minus, zeta_minus)
    dSA_db = (2*SA_plus - 2*SA_minus) / (2*h)
    g_A = np.abs(dSA_db) * norm_inv

    # --- TMDPDFs calculation with the n-framework NP model in the full-range of b_T --- #
    # here the definition with safe_eval

    
    f1_plus  = safe_eval( conv(x1_val, bt_plus, mub_plus,  zeta_plus), mub_plus >= mu_lower_limit and x1_val >= x_lower_limit )
    f1_minus = safe_eval( conv(x1_val, bt_minus, mub_minus, zeta_minus), mub_minus >= mu_lower_limit and x1_val >= x_lower_limit )
    f1_max   = safe_eval( conv(x1_val, bt_max, mub_max, zeta_max), mub_max >= mu_lower_limit and x1_val >= x_lower_limit )

    f2_plus  = safe_eval( conv(x2_val, bt_plus,  mub_plus,  zeta_plus), mub_plus >= mu_lower_limit and x2_val >= x_lower_limit )
    f2_minus = safe_eval( conv(x2_val, bt_minus, mub_minus, zeta_minus), mub_minus >= mu_lower_limit and x2_val >= x_lower_limit )
    f2_max   = safe_eval( conv(x2_val, bt_max, mub_max, zeta_max), mub_max >= mu_lower_limit and x2_val >= x_lower_limit  )

    """
    f1_plus  = conv(x1_val, bt_plus, mub_plus,  zeta_plus)
    f1_minus = conv(x1_val, bt_minus, mub_minus, zeta_minus)
    f1_max   = conv(x1_val, bt_max, mub_max, zeta_max)

    f2_plus  = conv(x2_val, bt_plus,  mub_plus,  zeta_plus)
    f2_minus = conv(x2_val, bt_minus, mub_minus, zeta_minus)
    f2_max   = conv(x2_val, bt_max, mub_max, zeta_max)
    """

    # avoid exact zero denominators (very unlikely but safe)
    denom1 = f1_max if abs(f1_max) > eps else eps
    denom2 = f2_max if abs(f2_max) > eps else eps

    # original: np.abs((f_plus - f_minus) / (2*h * f(bt_max))) * norm_inv
    g_f_1 = np.abs((f1_plus - f1_minus) / (2 * h * denom1)) * norm_inv
    g_f_2 = np.abs((f2_plus - f2_minus) / (2 * h * denom2)) * norm_inv

    # --- NP model with continuity constraint ---
    g_total = g_A + g_f_1 + g_f_2
    b_dagger2 = ( (bt**n + bmax**n)**(2 / n) ) - (bmax**2)
    SNP_bt = np.exp(- g_total * b_dagger2)

    # --- final b_star and scales for running evaluation ---
    b_star_value = b_star(bt, Q_sv, bmax)
    mu_bstar = c1 * (b0 / b_star_value)
    zeta_mubstar = c2**2 * (b0 / b_star_value)**2

    # here the definition with safe_eval
    F1 = safe_eval( conv(x1_val, b_star_value, mu_bstar, zeta_mubstar), mu_bstar >= mu_lower_limit and x1_val >= x_lower_limit )
    F2 = safe_eval( conv(x2_val, b_star_value, mu_bstar, zeta_mubstar), mu_bstar >= mu_lower_limit and x2_val >= x_lower_limit )

    """
    F1 = conv(x1_val, b_star_value, mu_bstar, zeta_mubstar)
    F2 = conv(x2_val, b_star_value, mu_bstar, zeta_mubstar)
    """

    SA_bstar = SA_f1g_nll_improved(b_star_value, Q_sv, Q_sv**2, mu_bstar, zeta_mubstar)

    return F1 * F2 * SNP_bt * np.exp(-2 * SA_bstar)


# -------------------------------------------------------
# Integatrion on y for LHCb kinematics
# -------------------------------------------------------


def integrand_W_nll_integrated_y(bt,Q,c1,c2, bmax):

    ymax = 4.5
    ymin = 2
    bin_width = ymax - ymin

    y_int = integrate.quad(lambda y: integrand_W_nll(bt,y,Q,c1,c2, bmax), ymin, ymax, epsrel = 1e-3, limit = 50)[0]

    return (1/(2*np.pi)) * y_int / bin_width

# -------------------------------------------------------
# Definition of the cross-section as qT * C[f1g f1g Sh]
# The NLO anomalous dimensions of the TMDShF is zero
# -------------------------------------------------------

# CHOOSE one of the two following definitions 

# In the following definition I use the Hankel defined function od mcfit 0.0.22
def unbinned_func(qt_array, Q, c1, c2, bmax):

    # 1. b-grid
    blow = 1e-10
    bhigh = 20
    N = 500
    b_grid = np.logspace(np.log10(blow), np.log10(bhigh), N)

    # 2. Compute W(b)
    Wb = np.array([
        integrand_W_nll_integrated_y(b, Q, c1, c2, bmax) for b in b_grid
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
    f_interp = np.interp(qt_array, q_out, f_q, left=0, right=0)

    # 6. Normalize
    norm = np.trapz(f_interp, qt_array)
    if norm > 0:
        f_interp /= norm
    
    return f_interp

""""
# In the following, I use the explicity definition of a Hankel Fourier Transform
def unbinned_func(qt_array, Q, c1, c2):

    bt_lim = 20

    def f1gf1g_qt(kt):
        def ft_f1gf1g_bt_integrand(bt, kt):
            return bt * special.jv(0, bt*kt, out = None) * integrand_W_nll_integrated_y(bt,Q,c1,c2)
        result = integrate.quad( lambda bt: ft_f1gf1g_bt_integrand(bt,kt), 0, bt_lim, epsrel = 1e-2, limit = 100)
        return result[0]
    
    def cs_value(x_value):
        return x_value * f1gf1g_qt(x_value)
    
    norm = integrate.quad(lambda kt: kt * f1gf1g_qt(kt), 0, qt_array[-1], epsrel = 1e-2, limit = 100)[0]

    cs_value_vect = np.vectorize(cs_value)
    
    return cs_value_vect(qt_array)/norm
"""

# -------------------------------------------------------
# Definition of the function which I want to compute.
# -------------------------------------------------------
def compute_one(params):

    start = time.time()

    Q, bmax = params
    print(f"Computing for Q={Q}, bmax={bmax}")

    x_max_map = {6.6: 3, 7.9: 4, 11: 6}
    x_max = x_max_map[Q]

    x = np.linspace(0, x_max, 5)
    y = unbinned_func(x, Q, 1, 1, bmax) # here we consider that c1 = c2

    end = time.time()

    print(y)
    print(end - start)

    return (Q, bmax, y)


# --------------------------------------------------
# Main program that runs everything in parallel in macOS
# --------------------------------------------------

def main():
    Q_values = [6.6, 7.9, 11] # here we consider only for one of the bins
    bmax_values = [0.75, 1, 1.5]

    start = time.time()

    # Build all parameter tuples
    params_list = list(itertools.product(Q_values, bmax_values))

    # Use all available CPU cores
    with mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
        results = pool.map(compute_one, params_list)

    # Store results as a dictionary for later notebook access
    result_dict = {}
    for Q, bmax, y in results:
        result_dict[(Q, bmax)] = y

    # Save with pickle
    with open("bmax_election_n10_MSHT20lo_as130.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    end = time.time()

    print(f"Execution time: {end - start:.4f} seconds")
    print("All results computed and saved to bmax_election_n10_MSHT20lo_as130.pkl")


if __name__ == "__main__":
    main()