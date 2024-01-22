import numpy as np
import numpy.linalg as LA
import scipy.io
import warnings
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from examples.twin_multi import twin_multi
from function_test.calculate_mean_square_error import calculate_mean_square_error
from pyoscdecomp.osc_ll_hess import osc_ll_hess
from pyoscdecomp.plot.osc_plot import osc_plot
from pyoscdecomp.plot.osc_phase_plot import osc_phase_plot
from pyoscdecomp.plot.osc_spectrum_plot import osc_spectrum_plot

# テスト 用の twin_multi の乱数の種です。
# 5489 または 1337 に対応しています。
seed = 5489
#seed = 1337

# MATLAB の twin_multi の入出力値を読み込みます。
filename = (
    parent_dir 
    + '/matlab_test_outputs/twin_multi{}.mat'.format(seed))
data = scipy.io.loadmat(filename)
osc_AIC_m = np.array(data['osc_AIC'])
osc_mean_m = np.array(data['osc_mean'])
osc_cov_m = np.array(data['osc_cov'])
osc_phase_m = np.array(data['osc_phase'])
minAIC_m = np.array(data['minAIC']).item()
K_m = np.array(data['K']).item()
osc_a_m = np.array(data['osc_a'])
osc_f_m = np.array(data['osc_f'])
osc_sigma2_m = np.array(data['osc_sigma2'])
osc_c_m = np.array(data['osc_c'])
osc_tau2_m = np.array(data['osc_tau2']).item()
hess_m = np.array(data['hess'])
grad_m = np.array(data['grad'])
mll_m = np.array(data['mll']).item()
cov_est_m = np.array(data['cov_est'])

# osc_mean,osc_cov, osc_phase の範囲は、AIC 最小の K に制限します。
osc_mean_K_m = osc_mean_m[0:2*K_m, :, K_m-1]
osc_cov_K_m = osc_cov_m[0:2*K_m, 0:2*K_m, :, K_m-1]
osc_phase_K_m = osc_phase_m[0:K_m, :, K_m-1]

# Python の twin_multi を実行します。
[osc_AIC, osc_mean, osc_cov, osc_phase,
 minAIC, K, osc_a, osc_f, osc_sigma2, osc_c, osc_tau2,
 Y, fs] = twin_multi(
    if_test=True, seed=seed)

# 信頼区間を計算します。
J = Y.shape[0]
osc_param = np.zeros((1, (2*J+1)*K+1))
osc_param[0:1, 0:(2*J+1)*K+1] = np.block(
    [osc_a, osc_f, osc_sigma2, osc_c, np.reshape(osc_tau2, (1, 1))])
hess, grad, mll = osc_ll_hess(Y, fs, osc_param[0:1, 0:(2*J+1)*K + 1])
cov_est = LA.inv(hess)
print('The number of oscillators is K={}'.format(K))
print('The periods of K oscillators are:')
for k in range(K):
    period = 1/osc_f[0, k]
    low = (1/(osc_f[0, k] + 1.96 * np.sqrt(cov_est[K+k, K+k].astype(complex)))).real
    up = (1/(osc_f[0, k] - 1.96 * np.sqrt(cov_est[K+k, K+k].astype(complex)))).real
    if (cov_est[K+k, K+k] < 0 
        or cov_est[K+k, K+k] > (osc_f[0, k]/1.96)**2):
        warnings.warn(
            'Confidence interval of the'
            + '{}-th oscilator period is complex or crosses zero.'.format(k+1)
        )
    print(
        ' {period:.2f} (95%% CI: [{low:.2f} {up:.2f}]) years\n'.format(
            period=period, low=low, up=up))

print('The phase differences for K oscillators correspond to:\n')
for k in range(K):
    phase_diff = np.arctan2(osc_c[0, 2*k+1], osc_c[0, 2*k])
    tmp = np.block([[-osc_c[0, 2*k+1]], [osc_c[0, 2*k]]]) / (osc_c[0, 2*k+1]**2 + osc_c[0, 2*k]**2)
    phase_var_est = tmp.T @ cov_est[3*K+2*k:3*K+2*k+2, 3*K+2*k:3*K+2*k+2] @ tmp
    period = phase_diff / 2 / np.pi / osc_f[0, k]
    low = ((phase_diff - 1.96 * np.sqrt(phase_var_est.astype(complex)).item()) / 2 / np.pi / osc_f[0, k]).real
    up = ((phase_diff + 1.96 * np.sqrt(phase_var_est.astype(complex)).item()) / 2 / np.pi / osc_f[0, k]).real
    if (phase_var_est < 0 
        or phase_var_est > (phase_diff/1.96)**2):
        warnings.warn(
            'Confidence interval of the'
            + '{}-th oscilator phase difference is complex or crosses zero.'.format(k+1)
        )
    print(
        ' {period:.2f} (95%% CI: [{low:.2f} {up:.2f}]) years\n'.format(
            period=period, low=low, up=up))

# 図示します。
osc_plot(osc_mean, osc_cov, fs, K)
osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K)
osc_spectrum_plot(Y, fs, osc_a, osc_f, osc_sigma2, osc_tau2, osc_c)

# osc_mean,osc_cov, osc_phase の範囲は、AIC 最小の K に制限します。
osc_mean_K = osc_mean[0:2*K, :, K-1]
osc_cov_K = osc_cov[0:2*K, 0:2*K, :, K-1]
osc_phase_K = osc_phase[0:K, :, K-1]

# 比較する変数と名称のリストを作成します。
var_list = [
    osc_AIC, osc_a, osc_f, osc_sigma2, osc_c, osc_tau2,
    osc_mean_K, osc_cov_K, osc_phase_K,
    ]
var_list_m = [
    osc_AIC_m, osc_a_m, osc_f_m, osc_sigma2_m, osc_c_m, osc_tau2_m,
    osc_mean_K_m, osc_cov_K_m, osc_phase_K_m,
    ]
name_list = [
    'osc_AIC', 'osc_a', 'osc_f', 'osc_sigma2', 'osc_c', 'osc_tau2',
    'osc_mean', 'osc_cov', 'osc_phase'
    ]

# 変数の値を表示します。
np.set_printoptions(formatter={'float': '{:.5e}'.format})
for var, var_m, name in zip(var_list, var_list_m, name_list):
    print('Python result of {}:'.format(name), var)
    print('MATLAB result of {}:'.format(name), var_m)

# AIC 最小の K が同じ値であれば、変数の誤差を計算します。
if K==K_m:
    for var, var_m, name in zip(var_list, var_list_m, name_list):
        err_relative, err_absolute = calculate_mean_square_error(var, var_m)
        print('Square mean error of {}:'.format(name))
        print('relative {:.3e}:'.format(err_relative),
            'absolute {:.3e}:'.format(err_absolute))
