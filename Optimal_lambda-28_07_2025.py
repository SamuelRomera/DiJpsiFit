
####### This script minimize the chi2 with iMinuit for a range of lambda in the modified chi2 definition by the curvature penalty.
####### The output is a table with the values of lambda, the fitted parameter, the chi2, and the corresponding penalty term.


import lhapdf
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import multiprocessing as mp
import os
import time
import csv

# Give the range of lambda
lam_initial, lam_final, n_lams = -3, 3, 100 # in log-scale

##################################################################################################
##################################################################################################
##################################################################################################

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
n = 8
def b_prime(bt,Q):
    return (bt**n + (b0/Q)**n )**(1/n)
def b_star(bt,Q,bmax):
    return b_prime(bt,Q) / (1 + (b_prime(bt,Q)/bmax)**n )**(1/n)

# sets the bmax value
bmax = 0.75

# sets the flavour-scheme
nf = 4

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

# sets the gluon fraction momenta
s_sqrt = 13000
def x1(y,Q):
    return Q * np.exp(y) / s_sqrt
def x2(y,Q):
    return Q * np.exp(-y) / s_sqrt

# sets the non-perturbative model for two gluons
def SNP(bt,Q,params,bmax):
    QNP = 1.6
    A = params[0]
    g1 = A * np.log(Q/QNP) 
    return np.exp( - g1 * ((bt**n + bmax**n)**(2/n) - bmax**2) ) 
    
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

# sets the LHAPDF

p = None

def init_worker(pdf_name = "MSHT20lo_as130"):
    global p
    print(f"[Init] Loading LHAPDF in PID {os.getpid()}")
    p = lhapdf.getPDFSet(pdf_name).mkPDF(0)

# sets the fit function

ndata = len(y1_norm_data) + len(y2_norm_data) + len(y3_norm_data)
nparameters = 1

initial_param = 0.414  # scalar, not array

def optimal_lambda(lam):

    global p
    print(f"[Run] λ = {lam:.4f} using PID {os.getpid()} and p = {p}")
    
    def integrand_W_nll(bt,y,Q,params,bmax):

        b_star_value = b_star(bt,Q,bmax)
        
        def As(Q):
            return p.alphasQ(Q)
        def a_s(Q):
            return As(Q)/(4*np.pi)

        def Conv_Cggf1g_delta_LO(x,bt,mui,zetai):
                return (p.xfxQ(0,x,mui)/x)
        def Conv_sumCgf1g_LO(x,bt,mui,zetai):
            return Conv_Cggf1g_delta_LO(x,bt,mui,zetai)
    
        # sets the perturbative Sudakov factor for one gluon
        LT = lambda bt,mu: np.log((mu**2) * (bt**2) / (b0**2))

        def SA_f1g_nll(bt,mu,zeta,mub,zeta0):
            def integrand(mup):
                return (1/mup) * ( ( a_s(mup) * Gammacusp0_g(nf) + (a_s(mup)**2) * Gammacusp1_g(nf)) * np.log(zeta/mup**2) + a_s(mup) * gammaV1_g(nf) )
            SA_nll_integral = integrate.quad( lambda mup: integrand(mup), mub, mu, epsrel = 1e-3, limit = 100 )[0]
            return ( (d10_g(nf) + d11_g(nf) * LT(bt,mub)) * a_s(mub) ) * np.log(zeta/zeta0) + SA_nll_integral

        return Conv_sumCgf1g_LO(x1(y,Q), b_star_value, b0/b_star_value, (b0/b_star_value)**2) * Conv_sumCgf1g_LO(x2(y,Q), b_star_value, b0/b_star_value, (b0/b_star_value)**2) * SNP(bt,Q,params,bmax) * np.exp( - 2 * SA_f1g_nll( b_star_value, Q, Q**2, b0/b_star_value, (b0/b_star_value)**2 ) )
    

    def fit_binned_func(x_data, Q, params, func):

        ymin = 2.0 # LHCb
        ymax = 4.5
        bt_lim = 20.0
        n_points_bt = 100

        def integrand_convolution(bt):
            y_int = integrate.quad(lambda y: func(bt,y,Q,params,bmax), ymin, ymax, epsrel = 1e-3, limit = 100)[0]
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
        sigma = np.sum(y_th)

        return y_th / sigma
    
    def curvature_penalty_func(Q, params, func): # this is only valid if x is a np.linspace()

        ktlim = Q
        ymin = 2.0 # LHCb
        ymax = 4.5
        bt_lim = 20.0
        n_points_bt = 100
        n_points_kt = 100

        def integrand_convolution(bt):
            y_int = integrate.quad(lambda y: func(bt,y,Q,params,bmax), ymin, ymax, epsrel = 1e-3, limit = 100)[0]
            return (bt/(2*np.pi)) * y_int / (ymax - ymin)
        integrand_convolution_vect = np.vectorize(integrand_convolution)

        bt_values = np.linspace(0.0, bt_lim, n_points_bt)
        interp_integrand_conv = interp1d(bt_values, integrand_convolution_vect(bt_values), kind='cubic')

        def f1gf1g_qt(kt):
            def ft_f1gf1g_bt_integrand(bt, kt):
                return special.jv(0, bt*kt, out = None) * interp_integrand_conv(bt)
            result = integrate.quad( lambda bt: ft_f1gf1g_bt_integrand(bt,kt), 0.0, bt_lim, epsrel = 1e-2, limit = 100)
            return result[0]
    
        f1gf1g_qt_vect = np.vectorize(f1gf1g_qt)
        kt_values = np.linspace(0.0, ktlim, n_points_kt)
        interp_FT = interp1d(kt_values, f1gf1g_qt_vect(kt_values), kind='cubic')
        norm = integrate.quad(lambda kt: kt * interp_FT(kt), 0.0, ktlim, epsrel = 1e-2, limit = 100)[0]

        def second_derivative(kt):
            def second_derivative_integrand(bt, kt):
                return (- (bt**2) * special.jv(0, bt*kt, out = None) - (bt/kt) * special.jv(1, bt*kt, out = None)) * interp_integrand_conv(bt)
            result = integrate.quad( lambda bt: second_derivative_integrand(bt,kt), 0.0, bt_lim, epsrel = 1e-2, limit = 100)
            return result[0]
    
        second_derivative_vect = np.vectorize(second_derivative)
        kt_values = np.linspace(1e-6, ktlim, n_points_kt)
        interp_second_derivative = interp1d(kt_values, second_derivative_vect(kt_values), kind='cubic')
    
    
        penalty_term = integrate.quad(lambda kt: (kt * interp_second_derivative(kt) / norm)**2, 0.0, ktlim, epsrel = 1e-3, limit = 100)[0]

        return penalty_term
    

    def chi2_data(params, func):

        chisq = 0

        for x_data, y_norm_data, inv_cov, Q in [ (x1_data, y1_norm_data, inv_cov_x1, 6.6), (x2_data, y2_norm_data, inv_cov_x2, 7.9), (x3_data, y3_norm_data, inv_cov_x3, 11.0)]:
        
            model = fit_binned_func(x_data, Q, [params[0]], func)
            delta = y_norm_data - model
            chisq += np.einsum("i,j,ij", delta, delta, inv_cov)

        return chisq

    def chisq_with_penalty(params, lam, func, return_components):

        Q_penalty = 3
        chisq_data_value = chi2_data(params, func)
        curvature_penalty_value = curvature_penalty_func(Q_penalty, params, func)

        chisq_total = chisq_data_value + lam * curvature_penalty_value

        
        if return_components:
            return chisq_total, chisq_data_value, curvature_penalty_value
    
        return chisq_total

    res = minimize(lambda A: chisq_with_penalty([A], lam, integrand_W_nll, return_components=False), initial_param, bounds=[(0.0, 20.0)], method="L-BFGS-B")

    # Get best-fit parameters
    best_params = res.x[0]  # keep as list to match chisq_2_modified
    chisq_total_value, chisq_data_value, curvature_penalty_value = chisq_with_penalty([best_params], lam, integrand_W_nll, return_components=True)
    
    return (lam, best_params, chisq_data_value / (ndata-nparameters), curvature_penalty_value)

# sets the Parallel execution

def main():
    start = time.time()
    lam_values = np.logspace(lam_initial, lam_final, n_lams)

    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:
        results = pool.map(optimal_lambda, lam_values)

    for r in results:
        print(f"λ = {r[0]:.4f} | Best A = {r[1]:.4f} | χ²data/ndof = {r[2]:.4f} | Curvature_penalty = {r[3]:.4f}")
    
    with open("Finding_Optimal_lambda_NLL.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "A", "chi2_dalta", "Curvature_penalty"])
        writer.writerows(results)

    print(f"\n✅ Results saved to 'Finding_Optimal_lambda_NLL.csv'")
    print(f"\nTotal time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()