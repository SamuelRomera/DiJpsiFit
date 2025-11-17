
####### This script calculate the corresponding 27 chi2s in terms of the 27 possible scale variations.
####### It is not a minimization of chi2, the NP parameters are fixed by the continuity constraint.


import lhapdf
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import interp1d
import os
from concurrent.futures import ProcessPoolExecutor
import itertools
import time
import csv


####### Give a b*-prescription and a level of accuracy (this is explicitily defined in the definition of the NP model in the code)
####### Current accuracy : NNLL

laccuracy = 'NNLL'

# sets the bT-prescription
n = 2
def b_prime(bt,Q):
    return (bt**n + (b0/Q)**n )**(1/n)
def b_star(bt,Q,bmax):
    return b_prime(bt,Q) / (1 + (b_prime(bt,Q)/bmax)**n )**(1/n)


##################################################################################################
##################################################################################################
##################################################################################################

# sets the flavour-scheme
nf = 4

# sets the LHAPDF
p = None

def init_worker(pdf_name = "MSHT20lo_as130"):
    global p
    print(f"[Init] Loading LHAPDF in PID {os.getpid()}")
    p = lhapdf.getPDFSet(pdf_name).mkPDF(0)


# define alpha_s
def As(Q):
    return p.alphasQ(Q)
def a_s(Q):
    return As(Q)/(4*np.pi)


# sets the gluon fraction momenta
s_sqrt = 13000
def x1(y,Q):
    return Q * np.exp(y) / s_sqrt
def x2(y,Q):
    return Q * np.exp(-y) / s_sqrt

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


# At this point, we can define the so-called g_A (as the absolute value of the derivative of the sudakov factor at b=bmax)
# It is needed to change the definition depending on the level of accuracy desired

def g_A(Q,bmax,n):

    h = 10**(-4)

    bt_plus = bmax + h
    bt_minos = bmax - h
    mub_plus = b0/bt_plus
    mub_minos = b0/bt_minos
    zeta0_plus = mub_plus**2
    zeta0_minos = mub_minos**2

    numerator = np.abs( (2 * SA_f1g_nnll(bt_plus,Q,Q**2,mub_plus,zeta0_plus) - 2 * SA_f1g_nnll(bt_minos,Q,Q**2,mub_minos,zeta0_minos)) / (2 * h) )
    return (numerator) / (2**(2/n) * bmax)


# sets the WL matching coefficients at LO and NLO
def Conv_sumCgf1g_NLO(x,bt,mui,zetai):

    def Conv_Cggf1g_delta(x,bt,mui,zetai):
        return (p.xfxQ(0,x,mui)/x) * (1 + (As(mui) / (2*np.pi)) * (CA * (- LT(bt,mui)**2/2 + LT(bt,mui) * np.log(mui**2/zetai) - np.pi**2/12)) )

    def Conv_Cggf1g_plus(x,bt,mui,zetai):
        def integrand(xh):
            return (As(mui)/(2*np.pi)) * (-LT(bt,mui)) * 2 * CA * (1/(1-xh)) * ( (p.xfxQ(0,x/xh,mui)/(x/xh)) - (p.xfxQ(0,x,mui)/x) )
        result = integrate.quad(lambda xh: integrand(xh), x, 1.0, epsrel=1e-4, limit=500)[0]
        return result + (As(mui)/(2*np.pi)) * (-LT(bt,mui)) * 2 * CA * (p.xfxQ(0,x,mui)/x) * np.log(1 - x)

    def Conv_Cggf1g_finite(x,bt,mui,zetai):
        def integrand(xh):
            return (1/xh) * (p.xfxQ(0,x/xh,mui)/(x/xh)) * (As(mui) / (2*np.pi)) * (- LT(bt,mui)) * 2 * CA * ( (1-xh)/(xh) + xh*(1-xh) )
        result = integrate.quad(lambda xh: integrand(xh), x, 1.0, epsrel=1e-4, limit=500)[0]
        return result

    def Conv_Cgqf1g(x,bt,mui,zetai):

        def sumQuarkPDF(xh,mui):
            sum = 0
            flavors = np.concatenate((np.arange(-4, 0), np.arange(1, 5))) # quark + antiquark
            for i in flavors:
                sum = sum + (p.xfxQ(i,x/xh,mui)/(x/xh))
            return sum
    
        def integrand(xh):
            return (1/xh) * sumQuarkPDF(xh,mui) * (As(mui)/(2*np.pi)) * (-LT(bt,mui) * CF * (1+ (1-xh)**2)/(xh) + CF * xh)
    
        result = integrate.quad(lambda xh: integrand(xh), x, 1.0, epsrel=1e-4, limit=500)[0]

        return result
    
    return Conv_Cggf1g_delta(x,bt,mui,zetai) + Conv_Cggf1g_plus(x,bt,mui,zetai) + Conv_Cggf1g_finite(x,bt,mui,zetai) + Conv_Cgqf1g(x,bt,mui,zetai)


def Conv_sumCgf1g_LO(x,bt,mui,zetai):
    return (p.xfxQ(0,x,mui)/x)

# At this point, we can define the so-called g_f for one gluon (as the absolute value of the derivative of ln(perturbative TMD) at b=bmax)
# It is needed to change the definition depending on the level of accuracy desired

def g_f(x,bmax,n):

    h = 10**(-4)

    bt_plus = bmax + h
    bt_minos = bmax - h
    mub_plus = b0/bt_plus
    mub_minos = b0/bt_minos
    zeta0_plus = mub_plus**2
    zeta0_minos = mub_minos**2

    numerator = np.abs( ( np.log(Conv_sumCgf1g_NLO(x,bt_plus,mub_plus,zeta0_plus)) - np.log(Conv_sumCgf1g_NLO(x,bt_minos,mub_minos,zeta0_minos)) ) / (2 * h) )
    return (numerator) / (2**(2/n) * bmax)

# At this stage, we can define the NP model with continuity constraint

def SNP(bt,y,Q,bmax,n):

    g = g_A(Q,bmax,n) + g_f(x1(y,Q),bmax,n) + g_f(x2(y,Q),bmax,n)
    b_dagger_2 = ((bt**n + bmax**n)**(2/n) - bmax**2)

    return np.exp( - g * b_dagger_2 ) 


# sets the experimental data at LHCb
def y_total_err(stat, syst):
    return np.sqrt( np.square(stat) + np.square(syst) )
y_err_vect = np.vectorize(y_total_err)


x1_data = np.array([0.5, 1.5, 2.5])
y1_data_SPS = np.array([0.400, 0.475, 0.591])
y1_stat_uncer = np.array([0.067, 0.103, 0.146])
y1_syst_uncer = np.array([0.032, 0.055, 0.062])

x2_data = np.array([0.5, 1.5, 2.5, 3.5])
y2_data_SPS = np.array([0.376, 0.501, 0.424, 0.438])
y2_stat_uncer = np.array([0.064, 0.114, 0.131, 0.126])
y2_syst_uncer = np.array([0.038, 0.081, 0.093, 0.087])

x3_data = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
y3_data_SPS = np.array([0.109, 0.214, 0.221, 0.200, 0.156, 0.099, 0.102])
y3_stat_uncer = np.array([0.038, 0.070, 0.073, 0.081, 0.067, 0.054, 0.042])
y3_syst_uncer = np.array([0.015, 0.037, 0.048, 0.051, 0.046, 0.037, 0.028])

bin_edges_1 = np.array([0.0, 1.0, 2.0, 3.0])
bin_width_1 = np.diff(bin_edges_1)
bin_edges_2 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
bin_width_2 = np.diff(bin_edges_2)
bin_edges_3 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
bin_width_3 = np.diff(bin_edges_3)

sigma_1 = np.sum(y1_data_SPS * bin_width_1)
sigma_2 = np.sum(y2_data_SPS * bin_width_2)
sigma_3 = np.sum(y3_data_SPS * bin_width_3)

y1_norm_data = y1_data_SPS / sigma_1
y1_norm_stat_uncer = y1_stat_uncer / sigma_1
y1_norm_syst_uncer = y1_syst_uncer / sigma_1
y_errsq_1 = y_total_err(y1_stat_uncer, y1_syst_uncer) / sigma_1

y2_norm_data = y2_data_SPS / sigma_2
y2_norm_stat_uncer = y2_stat_uncer / sigma_2
y2_norm_syst_uncer = y2_syst_uncer / sigma_2
y_errsq_2 = y_total_err(y2_stat_uncer, y2_syst_uncer) / sigma_2

y3_norm_data = y3_data_SPS / sigma_3
y3_norm_stat_uncer = y3_stat_uncer / sigma_3
y3_norm_syst_uncer = y3_syst_uncer / sigma_3
y_errsq_3 = y_total_err(y3_stat_uncer, y3_syst_uncer) / sigma_3

correlation_number = 0.0 # cn in [0, 1],  fully correlated systematics --> 1.0

cov_sta_x1 = np.diag(np.square(y1_stat_uncer))
cov_sta_x2 = np.diag(np.square(y2_stat_uncer))
cov_sta_x3 = np.diag(np.square(y3_stat_uncer))

n1 = len(y1_syst_uncer)
n2 = len(y2_syst_uncer)
n3 = len(y3_syst_uncer)

cov_sys_x1 = np.zeros((n1, n1))
cov_sys_x2 = np.zeros((n2, n2))
cov_sys_x3 = np.zeros((n3, n3))

for i in range(n1):
    for j in range(n1):
        cov_sys_x1[i, j] = (
            y1_syst_uncer[i] * y1_syst_uncer[j] * correlation_number
            if i != j else y1_syst_uncer[i] ** 2
        )

for i in range(n2):
    for j in range(n2):
        cov_sys_x2[i, j] = (
            y2_syst_uncer[i] * y2_syst_uncer[j] * correlation_number
            if i != j else y2_syst_uncer[i] ** 2
        )

for i in range(n3):
    for j in range(n3):
        cov_sys_x3[i, j] = (
            y3_syst_uncer[i] * y3_syst_uncer[j] * correlation_number
            if i != j else y3_syst_uncer[i] ** 2
        )

cov_x1 = cov_sta_x1 + cov_sys_x1
cov_x2 = cov_sta_x2 + cov_sys_x2
cov_x3 = cov_sta_x3 + cov_sys_x3

# Normalize the covariance matrix
J1 = np.zeros((n1, n1))
J2 = np.zeros((n2, n2))
J3 = np.zeros((n3, n3))

for i in range(n1):
    for k in range(n1):
        J1[i, k] = ( (1.0 if i == k else 0.0) * sigma_1 - y1_data_SPS[i] ) / (sigma_1 * sigma_1)

for i in range(n2):
    for k in range(n2):
        J2[i, k] = ( (1.0 if i == k else 0.0) * sigma_2 - y2_data_SPS[i] ) / (sigma_2 * sigma_2)

for i in range(n3):
    for k in range(n3):
        J3[i, k] = ( (1.0 if i == k else 0.0) * sigma_3 - y3_data_SPS[i] ) / (sigma_3 * sigma_3)

cov_x1_norm = J1 @ cov_x1 @ J1.T
cov_x2_norm = J2 @ cov_x2 @ J2.T
cov_x3_norm = J3 @ cov_x3 @ J3.T

inv_cov_x1 = np.linalg.pinv(cov_x1_norm)
inv_cov_x2 = np.linalg.pinv(cov_x2_norm)
inv_cov_x3 = np.linalg.pinv(cov_x3_norm)


# sets the chi2 definition at different levels of accuracy for scale variation.
# Here the scale variation is defined by three parameters, each one associated with: 
# c2  ->  hard scale, c1  ->  TMDs, c3  ->  Evolution


def chi2_data(c1, c2, c3, bmax):

    global p
    print(f"[Run] using PID {os.getpid()} and p = {p}")

    def integrand_W_nnll(bt,y,Q,bmax):

        Q_sv = c2 * Q
        b_star_value = b_star(bt,Q_sv,bmax)
        mu_bstar_sv_SA = (c1 * b0)/b_star_value
        mu_bstar_sv_TMD = (c3 * b0)/b_star_value

        return Conv_sumCgf1g_NLO(x1(y,Q_sv), b_star_value, mu_bstar_sv_TMD, (mu_bstar_sv_TMD)**2) * Conv_sumCgf1g_NLO(x2(y,Q_sv), b_star_value, mu_bstar_sv_TMD, (mu_bstar_sv_TMD)**2) * SNP(bt,y,Q_sv,bmax,n) * np.exp( - 2 * SA_f1g_nnll( b_star_value, Q_sv, Q_sv**2, mu_bstar_sv_SA, (mu_bstar_sv_SA)**2 ) )


    # sets the binned function to fit to data
    def fit_binned_func(x_data, Q):
    
        ymin = 2.0 # LHCb
        ymax = 4.5
        bt_lim = 20.0
        n_points_bt = 100

        def integrand_convolution(bt):
            y_int = integrate.quad(lambda y: integrand_W_nnll(bt,y,Q,bmax), ymin, ymax, epsrel = 1e-3, limit = 100)[0]
            return (bt/(2*np.pi)) * y_int / (ymax - ymin)
    
        integrand_convolution_vect = np.vectorize(integrand_convolution)
        bt_values = np.linspace(0.0, bt_lim, n_points_bt)
        interp_integrand_conv = interp1d(bt_values, integrand_convolution_vect(bt_values), kind='cubic')

        def f1gf1g_qt(kt):
            def ft_f1gf1g_bt_integrand(bt, kt):
                return special.jv(0, bt*kt, out = None) * interp_integrand_conv(bt)
            result = integrate.quad( lambda bt: ft_f1gf1g_bt_integrand(bt,kt), 0.0, bt_lim, epsrel = 1e-3, limit = 100)
            return result[0]

        y_th = np.array([ integrate.quad( lambda kt: kt * f1gf1g_qt(kt), x_value - 0.5, x_value + 0.5, epsrel = 1e-3, limit = 100 )[0] for x_value in x_data ])
        norm = np.sum(y_th)

        return y_th / norm

    # sets the definition of the chi2 w/o penalty

    chisq = 0
    ndata = len(y1_norm_data) + len(y2_norm_data) + len(y3_norm_data)

    for x_data, y_norm_data, inv_cov, Q in [ (x1_data, y1_norm_data, inv_cov_x1, 6.6), (x2_data, y2_norm_data, inv_cov_x2, 7.9), (x3_data, y3_norm_data, inv_cov_x3, 11.0)]:
        
        model = fit_binned_func(x_data, Q)
        delta = y_norm_data - model
        chisq += np.einsum("i,j,ij", delta, delta, inv_cov)

    return chisq / ndata


def main():

    start = time.time()

    c_values = [0.5, 1, 2]
    bmax_values = [0.75, 1, 1.5]

    # Generete all the combinations of c1,c2,c3 and bmax: 81 in total
    combinaciones = list(itertools.product(c_values, c_values, c_values, bmax_values))

    results = []

    # sets the Parallel execution
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        futuros = {
            executor.submit(chi2_data, c1, c2, c3, bmax): (c1, c2, c3, bmax)
            for (c1, c2, c3, bmax) in combinaciones
        }

        results = []

        # Restore the results
        for futuro in futuros:
            c1, c2, c3, bmax = futuros[futuro]
            valor = futuro.result()

            results.append(((c1, c2, c3), bmax, valor))

    # Show the results
    for c_tuple, bmax, val in results:
        print(f"{c_tuple} , bmax={bmax}  --->  {val}")

    # Save to CSV
    with open("chi2_SV_wcontinuity_SV_NNLL.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["(c1,c2,c3)", "bmax", "chi2_data"])
        writer.writerows(results)
    
    print(f"\n✅ Results for n = {n} at {laccuracy} saved to 'chi2_SV_wcontinuity_SV_NNLL.csv'")
    print(f"\nTotal time: {time.time() - start:.2f} seconds")

    return results
    


if __name__ == "__main__":
    main()